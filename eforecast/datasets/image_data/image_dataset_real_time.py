import os
import traceback
import cv2
import numpy as np
import pandas as pd
import joblib
import torch
from einops import rearrange
from einops import repeat

from eforecast.datasets.files_manager import FilesManager


class ImageDatasetRealTime(torch.utils.data.Dataset):
    def __init__(self, static_data, data, target, dates, params, device, train=True, use_target=True, is_online=False,
                 api='eumetview'):
        self.y = None
        self.x = None
        self.use_target = use_target
        self.device = device
        self.train = train
        self.horizon = static_data['horizon']
        self.spatial_coord_use = params['spatal_coord']
        self.static_data = static_data
        self.final_size = int(params['final_image_size'])
        self.image_type = params['image_data_type'].split(':')
        self.type = static_data['type']
        self.lags = [var_data for var_data in static_data['variables'] if var_data['type'] == 'image'][0]['lags']
        self.ts_resolution = static_data['ts_resolution']
        self.path_sat_processed = static_data['path_image']
        if not is_online:
            files_manager = FilesManager(static_data, train=train, is_online=is_online)
            dates_image, _ = files_manager.check_if_exists_image_data()
            dates = dates.intersection(dates_image)
        self.dates = dates
        indices = dates.get_indexer(self.dates)

        self.init_data(data, target, indices)
        self.params = params
        self.static_data = static_data


    def init_data(self, data, target, indices):
        if data is not None:
            self.x = dict()
            if isinstance(data, dict):
                for name in data.keys():
                    if name == 'images':
                        continue
                    if isinstance(data[name], dict):
                        self.x[name] = dict()
                        for name1 in data[name].keys():
                            values = data[name][name1][indices] if isinstance(data[name][name1], np.ndarray) \
                                else data[name][name1].values[indices]
                            self.x[name][name1] = torch.from_numpy(values)
                    else:
                        values = data[name][indices] if isinstance(data[name], np.ndarray) else data[name].values[indices]
                        self.x[name] = torch.from_numpy(values)
            else:
                self.x['input'] = torch.from_numpy(data[indices])
        if self.train:
            self.y = torch.from_numpy(target[indices]) if target is not None else None
        else:
            self.y = None

    def get_spatial_coords(self, params, api):
        site_coord = params['coord']
        site_coord = np.expand_dims(np.array(site_coord), (1, 2))
        spatial_coord = params['image_coord'][api]
        image_size = params['image_size'][api]
        lat, long = self.static_data['site_indices'][api]
        area_adjust = params['area_adjust'][api]
        lat_grid = np.linspace(spatial_coord[0], spatial_coord[1], image_size[0])
        lon_grid = np.linspace(spatial_coord[2], spatial_coord[3], image_size[1])
        spatial_coords = np.stack(np.meshgrid(lon_grid, lat_grid)[::-1], axis=0)
        site_coord = repeat(site_coord, 'n w h -> n (w k) (h m)', k=spatial_coords.shape[1],
                            m=spatial_coords.shape[2])

        data = spatial_coords - site_coord
        data = np.power(data[0], 2) + np.power(data[1], 2)
        data = data[(None,) * 3 + (...,)]
        data = repeat(data, 'b t c w h -> b (t k) c w h', k=len(self.lags))
        spatial_coord_3d = data[:, :, :,
                           np.maximum(lat - area_adjust, 0):np.minimum(lat + area_adjust, data.shape[-2]),
                           np.maximum(long - area_adjust, 0):np.minimum(long + area_adjust, data.shape[-1])]
        spatial_coord_3d = spatial_coord_3d.squeeze()
        spatial_coord_3d = rearrange(spatial_coord_3d, 'c w h -> w h c')
        spatial_coord_3d = np.concatenate(
            [np.expand_dims(cv2.resize(spatial_coord_3d[:, :, i],
                                       dsize=[self.final_size, self.final_size],
                                       interpolation=cv2.INTER_AREA), axis=-1)
             for i in range(spatial_coord_3d.shape[-1])], -1)
        spatial_coord_3d = rearrange(spatial_coord_3d, 'w h c -> 1 1 w h c')
        spatial_coord_3d = self.final_resize(spatial_coord_3d)
        spatial_coord_3d = rearrange(spatial_coord_3d, 'b c w h t -> b t w h c')

        return spatial_coord_3d.astype(np.float32)

    def __len__(self) -> int:
        return self.dates.shape[0]

    def __getitem__(self, idx):
        try:
            return self.get(idx)
        except:
            return None, None

    def get_image_grey(self, images):
        inp_lag = []
        for j in range(images.shape[0]):
            sat = images[j, :, :, :]
            sat = np.expand_dims(cv2.cvtColor(sat.astype(np.float32), cv2.COLOR_BGR2GRAY), axis=-1)
            inp_lag.append(np.expand_dims(sat, axis=0))
        return np.expand_dims(np.concatenate(inp_lag, axis=0), axis=0)

    def final_resize(self, images):
        image_res0 = []
        for k in range(images.shape[0]):
            image_res2 = []
            for g in range(images.shape[1]):
                img_crop = np.concatenate(
                    [np.expand_dims(cv2.resize(images[k, g, :, :, i],
                                               dsize=[self.final_size, self.final_size],
                                               interpolation=cv2.INTER_AREA), axis=-1)
                     for i in range(images.shape[-1])], -1)
                image_res2.append(img_crop)
            image_res1 = np.array(image_res2)
            image_res0.append(image_res1)
        image = np.array(image_res0)
        return image

    def get_image(self, date):
        x = []
        api = self.image_type[0].split('_')[0]
        for img_tag in self.image_type:
            img_tag = '_'.join(img_tag.split('_')[1:])
            try:
                x_img = joblib.load(os.path.join(self.path_sat_processed, 'processed',
                                               f'satellite_{api}_{img_tag}_{date.strftime("%Y_%m_%d__%H_%M")}.pkl'))
            except:
                print('Something went wrong')
                raise
            x.append(x_img[img_tag])
        self.static_data.update(self.params)

        if len(x) == 1:
            x = x[0]
        else:
            x = np.concatenate(x, axis=-1)
        B, L, W, H, C = x.shape

        centre = W // 2
        a = self.static_data['area_adjust'][api]
        x = x[:, :, np.maximum(0, centre - a):np.minimum(centre + a, W),
            np.maximum(0, centre - a):np.minimum(centre + a, H), :]
        if self.final_size != W or self.final_size != H:
            x = self.final_resize(x)
        if self.spatial_coord_use:
            spatial_coords = self.get_spatial_coords(self.static_data, api)
            x = np.concatenate([x, spatial_coords], axis=-1)
        x = torch.from_numpy(x.astype(np.float32))
        return rearrange(x, 'b l w h c -> b l c w h')


    def get_data(self, idx):

        x_data = dict()
        if isinstance(self.x, dict):
            for name in self.x.keys():
                if isinstance(self.x[name], dict):
                    x_data[name] = dict()
                    for name1 in self.x[name].keys():
                        x_data[name][name1] = self.x[name][name1][idx].float().to(self.device)
                else:
                    x_data[name] = self.x[name][idx].float().to(self.device)
        else:
            raise ValueError('Input must be dict')
        return x_data

    def get(self, idx):
        date = self.dates[idx]
        try:
            if self.x is not None:
                X = self.get_data(idx)
            else:
                X = None
        except Exception as e:
            tb = traceback.format_exception(e)
            print("".join(tb))
            raise
        dates_obs = pd.DatetimeIndex([date + pd.DateOffset(hours=l) for l in self.lags][::-1])
        dates_pred = pd.date_range(date, date + pd.DateOffset(hours=self.horizon), freq=self.ts_resolution)
        try:
            x_img_obs = self.get_image(date)
        except Exception as e:
            tb = traceback.format_exception(e)
            if 'FileNotFoundError' not in "".join(tb):
                print("".join(tb))
            raise
        return_tensors = {
            "images": x_img_obs[0].float().to(self.device),
        }
        if X is not None:
            return_tensors.update(X)
        if self.use_target:
            try:
                x_img_pred = self.get_image_eumdac(dates_pred)
            except:
                print('Something went wrong')
                raise
            target = torch.from_numpy(x_img_pred).float()
        elif self.train and self.y is not None:
            target = self.y[idx].float().to(self.device)
        else:
            target = None
        if target is not None:
            return return_tensors, target
        else:
            return return_tensors, date
