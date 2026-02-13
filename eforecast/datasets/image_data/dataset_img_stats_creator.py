import os
import cv2
import rasterio
import h5py
import joblib
from joblib import Parallel
from joblib import delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
import astral
from astral.sun import sun
from einops import rearrange
from einops import repeat
import torch
import torch.nn as nn
from multiprocessing import Pool

from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.models import VAEConfig
from pythae.models import VAE

import segmentation_models_pytorch as smp
from sewar.full_ref import mse, ssim
from eforecast.common_utils.date_utils import convert_timezone_dates

class Encoder_Conv_VAE_MNIST(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]

        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 16, 4, 2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 4, 2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.embedding = nn.Linear(1024, args.latent_dim)
        self.log_var = nn.Linear(1024, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOutput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class Decoder_Conv_AE_MNIST(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = args.input_dim[0]

        self.fc = nn.Linear(args.latent_dim, 1024 * 5 * 5)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.n_channels, 4, 2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], 1024, 5, 5)
        output = ModelOutput(reconstruction=self.deconv_layers(h1))

        return output

def vae_model(x_shape=None, y_shape=None):
    model_config1 = VAEConfig(
        input_dim=x_shape,
        latent_dim=16,
    )
    model_config2 = VAEConfig(
        input_dim=y_shape,
        latent_dim=16,
    )
    encoder = Encoder_Conv_VAE_MNIST(model_config1)
    decoder = Decoder_Conv_AE_MNIST(model_config2)

    model = VAE(
        model_config=model_config1,
        encoder=encoder,
        decoder=decoder
    )
    return model

class DatasetImageStatsCreator:

    def __init__(self, static_data, transformer, dates=None, is_online=False, parallel=False, refit=False):
        self.refit = refit
        self.static_data = static_data
        self.transformer = transformer
        self.is_online = is_online
        self.parallel = parallel
        ts_res = str.lower(static_data['ts_resolution'])
        if self.is_online:
            self.dates = dates
        else:
            dates = dates.round(ts_res).unique()
            self.dates = self.remove_night_hours(dates)
        self.path_sat = static_data['sat_folder']
        self.path_sat_processed = static_data['path_image']
        self.n_jobs = static_data['n_jobs']
        self.variables = dict([(var_data['name'], var_data) for var_data in static_data['variables']
                               if var_data['type'] == 'image'])
        self.apis = set([var_data['source'] for var_data in self.variables.values()])
        for var in self.variables.keys():
            if var in self.transformer.variables_index.keys():
                self.transformer.fit(np.array([]), var, data_dates=dates)
        self.coord = static_data['coord']
        self.area_adjust = {'eumetdac': 50, 'eumetview': 80}
        print(f"Dataset Image Stats creation started for project {self.static_data['_id']}")

    def daylight(self, date):
        try:
            l = astral.LocationInfo('Custom Name', 'My Region', self.static_data['local_timezone'],
                                    self.static_data['coord'][0], self.static_data['coord'][1])
            sun_attr = sun(l.observer, date=date, tzinfo=self.static_data['local_timezone'])
            sunrise = pd.to_datetime(sun_attr['dawn'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
            sunset = pd.to_datetime(sun_attr['dusk'].strftime('%Y%m%d %H:%M'), format='%Y%m%d %H:%M')
            if sunrise - pd.DateOffset(hours=3) <= date <= sunset + pd.DateOffset(hours=3):
                return date
            else:
                return None
        except:
            return None

    def remove_night_hours(self, dates):
        try:
            with Pool(self.static_data['n_jobs']) as pool:
                daylight_dates = pool.map(self.daylight, dates)
        except:
            daylight_dates = [self.daylight(date) for date in dates]
        daylight_dates = [d for d in daylight_dates if d is not None]
        dates_new = pd.DatetimeIndex(daylight_dates)
        return dates_new

    def make_dataset(self):
        if not os.path.exists(os.path.join(self.path_sat_processed, 'processed')):
            os.makedirs(os.path.join(self.path_sat_processed, 'processed'))
        stats_api = dict()
        for api in self.apis:
            for var in self.variables.keys():
                if self.variables[var]['source'] != api:
                    continue
                if not self.parallel:
                    stats = []
                    for t in tqdm(self.dates):
                        if os.path.exists(os.path.join(self.path_sat_processed, 'processed',
                                                           f'satellite_{api}_{var}_{t.strftime("%Y_%m_%d__%H_%M")}.pkl')):
                            res = self.stack_hourly_sat(t, api, var)
                            if res is None:
                                continue
                            data = res
                            stats.append(data)
                        else:
                            continue
                else:
                    data = Parallel(n_jobs=18)(
                        delayed(self.stack_hourly_sat)(t, api, var) for t in tqdm(self.dates)
                                                                    if os.path.exists(os.path.join(self.path_sat_processed,
                                                                                                   'processed',
                                                                                                    f'satellite_{api}_{var}_{t.strftime("%Y_%m_%d__%H_%M")}.pkl')))
                    stats = [d for d in data if d is not None]
                if len(stats) > 0:
                    stats_api[f'{api}_{var}'] = pd.concat(stats)
        return stats_api

    def stack_hourly_sat(self, t, api, var):
        x_3d = self.create_stats(t, api, var)
        return x_3d

    def create_stats(self, date, api, variable):
        stats = []
        horizon = self.static_data['target_variable']['lags'] if self.static_data['type'] == 'multi-output' \
                                                                else [i for i in range(self.static_data['horizon'])]
        for h in horizon:
            df = self.compute_stats(date, api, variable, h + 1)
            if df is None:
                return None
            stats.append(df)

        return pd.concat(stats, axis=1)

    def final_resize(self, images, final_size):
        image_res0 = []
        for k in range(images.shape[0]):
            image_res2 = []
            for g in range(images.shape[1]):
                img_crop = np.concatenate(
                    [np.expand_dims(cv2.resize(images[k, g, :, :, i],
                                               dsize=[final_size, final_size],
                                               interpolation=cv2.INTER_CUBIC), axis=-1)
                     for i in range(images.shape[-1])], -1)
                image_res2.append(img_crop)
            image_res1 = np.array(image_res2)
            image_res0.append(image_res1)
        image = np.array(image_res0)
        return image


    def get_image_eumetview(self, date, band):
        x_img = joblib.load(os.path.join(self.path_sat_processed, 'processed',
                                           f'satellite_eumetview_{band}_{date.strftime("%Y_%m_%d__%H_%M")}.pkl'))
        x = x_img[band]
        B, L, W, H, C = x.shape
        centre = [106, 111]
        a = self.area_adjust['eumetview']
        x = x[:, :, np.maximum(0, centre[0] - a):np.minimum(centre[0] + a, W),
            np.maximum(0, centre[1] - a):np.minimum(centre[1] + a, H), :]
        x = torch.from_numpy(x.astype(np.float32))
        return rearrange(x, 'b l w h c -> b l c w h')

    def get_image_eumdac(self, date, band):
        x_img = joblib.load(os.path.join(self.path_sat_processed, 'processed',
                                             f'satellite_eumetdac_{band}_{date.strftime("%Y_%m_%d__%H_%M")}.pkl'))
        x = x_img[band]
        B, L, W, H, C = x.shape
        centre = [55, 75]
        a = self.area_adjust['eumetdac']
        x = x[:, :, np.maximum(0, centre[0] - a):np.minimum(centre[0] + a, W),
            np.maximum(0, centre[1] - a):np.minimum(centre[1] + a, H), :]
        x = self.final_resize(x, 160)
        x = torch.from_numpy(x.astype(np.float32))
        return rearrange(x, 'b l w h c -> b l c w h')


    def compute_stats(self, date, api, var, h):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_weights = torch.load(os.path.join(self.static_data['path_group'],
                                               f'best_ae_{var}_model_hor_{h}.pt'),
                                  weights_only=False,
                                  map_location=device)
        torch.cuda.empty_cache()
        try:
            if api == 'eumetdac':
                image = self.get_image_eumdac(date, var)
            else:
                image = self.get_image_eumetview(date, var)
        except:
            return None
        inp = rearrange(image, 'b t c w h -> b (t c) w h') / 255
        if inp.shape[1] != 15:
            return None
        net_model = vae_model(x_shape=inp.shape[1:], y_shape=[3, *inp.shape[2:]]).to(device)
        net_model.load_state_dict(best_weights)
        net_model.to(device)
        net_model.eval()

        encoder_out = net_model.encoder(inp.to(device))
        encoder_out = torch.cat([encoder_out.embedding, encoder_out.log_covariance], -1).detach().cpu().numpy()
        stats = pd.DataFrame(encoder_out, index=[date], columns=[f'enc_{i}_{var}_{h}' for i in range(encoder_out.shape[1])])

        return stats
