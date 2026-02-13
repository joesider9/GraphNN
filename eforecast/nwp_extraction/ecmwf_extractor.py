import shutil
import tarfile
import copy
import joblib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import datetime
import base64
import pandas as pd

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

if sys.platform == 'linux':
    import pygrib
else:
    import cfgrib


class DownLoader:
    """
    Downloads the ECMWF attachment for a given date from Gmail using the Gmail API.

    - Searches only emails with label 'ECMWF'
    - Searches by the expected subject
    - Saves attachment as:
        {path_nwp}/{year}/SIDERT{mmdd}00UTC.tgz
    """

    # Read-only is enough
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

    def __init__(self, date=None, path_nwp=None,
                 credentials_file=None, token_file=None):

        # ---------- Resolve paths for credentials ----------
        if sys.platform == 'linux':
            if not os.path.exists('/media/sider/data'):
                base_path = '/home/smartrue/Dropbox/current_codes/PycharmProjects/ECMWF_download'
            else:
                base_path = '/media/sider/data/Dropbox/current_codes/PycharmProjects/ECMWF_download'
        else:
            if os.path.exists('D:/'):
                base_path = 'D:/Dropbox/current_codes/PycharmProjects/ECMWF_download'
            else:
                base_path = 'C:/Dropbox/current_codes/PycharmProjects/ECMWF_download'

        # OAuth client secret (downloaded from Google Cloud console)
        if credentials_file is None:
            credentials_file = os.path.join(base_path, 'credentials.json')
        # Token generated after first OAuth run
        if token_file is None:
            token_file = os.path.join(base_path, 'token.json')

        if not os.path.exists(credentials_file):
            # allow local fallback for dev
            file = [file for file in os.listdir(base_path) if file.startswith('client_secret')][0]
            credentials_file = os.path.join(base_path, file)
            if not os.path.exists(credentials_file):
                raise ImportError('Cannot find Gmail API credentials.json')

        self.credentials_file = credentials_file
        self.token_file = token_file

        # ---------- Date / subject / filename ----------
        if date is None:
            self.date = pd.to_datetime(datetime.datetime.now().strftime('%d%m%y'),
                                       format='%d%m%y')
        else:
            self.date = pd.to_datetime(date)

        if path_nwp is None:
            path_nwp = os.getcwd()
        self.path_nwp = path_nwp

        year_dir = os.path.join(self.path_nwp, str(self.date.year))
        os.makedirs(year_dir, exist_ok=True)

        file_name = f"SIDERT{self.date.strftime('%m%d')}00UTC.tgz"
        self.filename = os.path.join(year_dir, file_name)

        self.subject = f"Real Time data {self.date.strftime('%Y-%m-%d')} 00UTC"

        # Lazy init; service created on first use
        self._service = None

    # ---------- Auth / service creation ----------
    def _get_service(self):
        if self._service is not None:
            return self._service

        creds = None
        # Load existing token if present
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)

        # Refresh / create token if needed
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file,
                                                                 self.SCOPES)
                creds = flow.run_local_server(port=8080, open_browser=True)
            # Save token for next runs
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())

        self._service = build('gmail', 'v1', credentials=creds)
        return self._service

    # ---------- Core logic ----------
    def download(self):
        """
        Search for the ECMWF email and download its attachment to self.filename.
        Returns True if successful, False otherwise.
        """
        try:
            service = self._get_service()

            # Search: label Ecmwf + exact subject
            # Note: q is a standard Gmail search query
            query = f'subject:"{self.subject}"'
            label_ids = ['Label_7963952810175448694']

            response = service.users().messages().list(
                userId='me',
                q=query,
                labelIds=label_ids,
                maxResults=10
            ).execute()

            messages = response.get('messages', [])
            if not messages:
                print("No matching emails found for subject:", self.subject)
                return False

            # Iterate over matching messages (usually you'd expect 1)
            for msg in messages:
                msg_id = msg['id']
                message = service.users().messages().get(
                    userId='me', id=msg_id, format='full'
                ).execute()

                payload = message.get('payload', {})
                parts = payload.get('parts', [])

                # Walk through payload parts to find attachments
                for part in parts:
                    filename = part.get('filename')
                    body = part.get('body', {})
                    attachment_id = body.get('attachmentId')

                    # Skip inline/empty parts
                    if not filename or not attachment_id:
                        continue
                    if os.path.basename(self.filename) != filename:
                        continue
                    # If you only expect a specific name, you could check here:
                    # if not filename.endswith('.tgz'): continue

                    print("Saving attachment to:", self.filename)

                    attachment = service.users().messages().attachments().get(
                        userId='me',
                        messageId=msg_id,
                        id=attachment_id
                    ).execute()

                    file_data = attachment.get('data')
                    if file_data is None:
                        continue

                    file_bytes = base64.urlsafe_b64decode(file_data.encode('UTF-8'))

                    # Ensure directory exists
                    os.makedirs(os.path.dirname(self.filename), exist_ok=True)

                    with open(self.filename, 'wb') as f:
                        f.write(file_bytes)

                    # Assume only one relevant attachment per message
                    return True

            print('Not able to download attachments for:', self.subject)
            return False

        except Exception as e:
            print('Error while downloading attachment for', self.subject)
            print(repr(e))
            return False

class EcmwfExtractor:

    def __init__(self, dates, path_nwp):
        self.dates_ts = dates.floor('D').unique()
        self.path_nwp = path_nwp
        if not os.path.exists(self.path_nwp):
            os.makedirs(self.path_nwp)

    def extract_pygrib1(self, date_of_measurement, file_name):

        # We get 48 hours forecasts. For every date available take the next 47 hourly predictions.

        nwps = dict()
        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        for dt in dates:
            nwps[dt.strftime('%d%m%y%H%M')] = dict()
        grb = pygrib.open(file_name)
        temp = []
        for i in range(1, grb.messages + 1):
            g = grb.message(i)
            if g.cfVarNameECMF == 'u100':
                var = 'Uwind'
            elif g.cfVarNameECMF == 'v100':
                var = 'Vwind'
            elif g.cfVarNameECMF == 't2m':
                var = 'Temperature'
            elif g.cfVarNameECMF == 'tcc':
                var = 'Cloud'
            elif g.cfVarNameECMF == 'ssrd':
                var = 'Flux'
            dt = dates[g.endStep].strftime('%d%m%y%H%M')
            data, lat, long = g.data()  # Each "message" corresponds to a specific line on Earth.
            if var == 'Flux':
                if len(temp) == 0:
                    temp.append(data)
                else:
                    t = copy.deepcopy(data)
                    data = data - temp[0]
                    temp[0] = copy.deepcopy(t)
            nwps[dt]['lat'] = lat
            nwps[dt]['long'] = long
            nwps[dt][var] = data
        grb.close()
        del grb
        for dt in nwps.keys():
            Uwind = nwps[dt]['Uwind']
            Vwind = nwps[dt]['Vwind']
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180
            nwps[dt]['WS'] = wspeed
            nwps[dt]['WD'] = wdir
        return nwps

    def extract_cfgrib1(self, file_name):
        nwps = dict()
        data = cfgrib.open_dataset(file_name)
        dates = pd.to_datetime(data.valid_time.data, format='%Y-%m-%d %H:%M:%S').strftime('%d%m%y%H%M')
        Uwind = data.u100.data
        Vwind = data.v100.data
        temperature = data.t2m.data
        cloud = data.tcc.data
        flux = data.ssrd.data
        lat = data.latitude.data
        long = data.longitude.data
        r2d = 45.0 / np.arctan(1.0)
        wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
        wdir = np.arctan2(Uwind, Vwind) * r2d + 180
        for i, dt in enumerate(dates):
            nwps[dt] = dict()
            nwps[dt]['lat'] = lat
            nwps[dt]['long'] = long
            nwps[dt]['Uwind'] = Uwind[i]
            nwps[dt]['Vwind'] = Vwind[i]
            nwps[dt]['WS'] = wspeed[i]
            nwps[dt]['WD'] = wdir[i]
            nwps[dt]['Temperature'] = temperature[i]
            nwps[dt]['Cloud'] = cloud[i]
            if i == 0:
                temp = copy.deepcopy(flux[i])
            elif i > 0:
                temp1 = copy.deepcopy(flux[i])
                flux[i] = flux[i] - temp
                temp = copy.deepcopy(temp1)
            nwps[dt]['Flux'] = flux[i]

        return nwps

    def extract_pygrib2(self, date_of_measurement, file_name):
        path_extract = os.path.join(self.path_nwp, 'extract/' + date_of_measurement.strftime('%d%m%y'))
        if not os.path.exists(path_extract):
            os.makedirs(path_extract)
        tar = tarfile.open(file_name)
        tar.extractall(path_extract)
        tar.close()
        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        nwps = dict()
        temp = []
        for j, dt in enumerate(dates):
            file = os.path.join(path_extract,
                                'E_H6S' + date_of_measurement.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(
                                    dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(path_extract, 'E_H6S' + date_of_measurement.strftime(
                    '%m%d') + '0000' + date_of_measurement.strftime('%m%d') + '00011')
                if not os.path.exists(file):
                    continue

            grb = pygrib.open(file)
            nwps[dt.strftime('%d%m%y%H%M')] = dict()
            for i in range(1, grb.messages + 1):
                g = grb.message(i)
                if g.cfVarNameECMF == 'u100':
                    var = 'Uwind'
                elif g.cfVarNameECMF == 'v100':
                    var = 'Vwind'
                elif g.cfVarNameECMF == 't2m':
                    var = 'Temperature'
                elif g.cfVarNameECMF == 'tcc':
                    var = 'Cloud'
                elif g.cfVarNameECMF == 'ssrd':
                    var = 'Flux'

                data, lat, long = g.data()
                if var == 'Flux':
                    if len(temp) == 0:
                        temp.append(data)
                    else:
                        t = copy.deepcopy(data)
                        data = data - temp[0]
                        temp[0] = copy.deepcopy(t)
                nwps[dt.strftime('%d%m%y%H%M')]['lat'] = lat
                nwps[dt.strftime('%d%m%y%H%M')]['long'] = long
                nwps[dt.strftime('%d%m%y%H%M')][var] = data
            grb.close()
            del grb
        for dt in nwps.keys():
            try:
                if 'Uwind' not in nwps[dt].keys():
                    continue
                Uwind = nwps[dt]['Uwind']
                Vwind = nwps[dt]['Vwind']
                r2d = 45.0 / np.arctan(1.0)
                wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
                wdir = np.arctan2(Uwind, Vwind) * r2d + 180
                nwps[dt]['WS'] = wspeed
                nwps[dt]['WD'] = wdir
            except:
                continue
        return nwps

    def extract_cfgrib2(self, date_of_measurement, file_name):
        path_extract = os.path.join(self.path_nwp, 'extract/' + date_of_measurement.strftime('%d%m%y'))
        if not os.path.exists(path_extract):
            os.makedirs(path_extract)
        else:
            shutil.rmtree(path_extract)
            if not os.path.exists(path_extract):
                os.makedirs(path_extract)
        tar = tarfile.open(file_name)
        tar.extractall(path_extract)
        tar.close()
        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        nwps = dict()
        temp = []
        for i, dt in enumerate(dates):
            file = os.path.join(path_extract,
                                'E_H6S' + date_of_measurement.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(
                                    dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(path_extract, 'E_H6S' + date_of_measurement.strftime(
                    '%m%d') + '0000' + date_of_measurement.strftime('%m%d') + '00011')
                if not os.path.exists(file):
                    continue
            try:
                data = cfgrib.open_dataset(file)
            except:
                continue
            try:
                Uwind = data.u100.data
                Vwind = data.v100.data
                temperature = data.t2m.data
                cloud = data.tcc.data
                flux = data.ssrd.data
            except:
                continue
            if len(temp) == 0:
                temp.append(flux)
            else:
                t = copy.deepcopy(flux)
                flux = flux - temp[0]
                temp[0] = copy.deepcopy(t)
            lat = data.latitude.data
            long = data.longitude.data
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180

            nwp = dict()
            nwp['lat'] = lat
            nwp['long'] = long
            nwp['Uwind'] = Uwind
            nwp['Vwind'] = Vwind
            nwp['WS'] = wspeed
            nwp['WD'] = wdir
            nwp['Temperature'] = temperature
            nwp['Cloud'] = cloud
            nwp['Flux'] = flux
            nwps[dt.strftime('%d%m%y%H%M')] = nwp
        return nwps

    def extract_pygrib3(self, date_of_measurement, file_name):

        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        nwps = dict()
        temp = []
        for j, dt in enumerate(dates):
            file = os.path.join(file_name,
                                'E_H6S' + date_of_measurement.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(
                                    dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(file_name, 'E_H6S' + date_of_measurement.strftime(
                    '%m%d') + '0000' + date_of_measurement.strftime('%m%d') + '00011')

                if not os.path.exists(file):
                    continue

            grb = pygrib.open(file)
            nwps[dt.strftime('%d%m%y%H%M')] = dict()
            for i in range(1, grb.messages + 1):
                g = grb.message(i)
                if g.cfVarNameECMF == 'u100':
                    var = 'Uwind'
                elif g.cfVarNameECMF == 'v100':
                    var = 'Vwind'
                elif g.cfVarNameECMF == 't2m':
                    var = 'Temperature'
                elif g.cfVarNameECMF == 'tcc':
                    var = 'Cloud'
                elif g.cfVarNameECMF == 'ssrd':
                    var = 'Flux'

                data, lat, long = g.data()
                if var == 'Flux':
                    if len(temp) == 0:
                        temp.append(data)
                    else:
                        t = copy.deepcopy(data)
                        data = data - temp[0]
                        temp[0] = copy.deepcopy(t)
                nwps[dt.strftime('%d%m%y%H%M')]['lat'] = lat
                nwps[dt.strftime('%d%m%y%H%M')]['long'] = long
                nwps[dt.strftime('%d%m%y%H%M')][var] = data
            grb.close()
            del grb
        for dt in nwps.keys():
            Uwind = nwps[dt]['Uwind']
            Vwind = nwps[dt]['Vwind']
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180
            nwps[dt.strftime('%d%m%y%H%M')]['WS'] = wspeed
            nwps[dt.strftime('%d%m%y%H%M')]['WD'] = wdir
        return nwps

    def extract_cfgrib3(self, date_of_measurement, file_name):

        dates = pd.date_range(start=date_of_measurement, end=date_of_measurement + pd.DateOffset(hours=48), freq='h')
        nwps = dict()
        temp1 = []
        for i, dt in enumerate(dates):
            file = os.path.join(file_name,
                                'H6S' + date_of_measurement.strftime('%m%d') + '0000' + dt.strftime('%m%d') + str(
                                    dt.hour).zfill(
                                    2) + '001')
            if not os.path.exists(file):
                file = os.path.join(file_name, 'H6S' + date_of_measurement.strftime(
                    '%m%d') + '0000' + date_of_measurement.strftime('%m%d') + '00011')

                if not os.path.exists(file):
                    continue

            data = cfgrib.open_dataset(file)
            Uwind = data.u100.data
            Vwind = data.v100.data
            temp = data.t2m.data
            cloud = data.tcc.data
            flux = data.ssrd.data
            if len(temp1) == 0:
                temp1.append(flux)
            else:
                t = copy.deepcopy(flux)
                flux = flux - temp1[0]
                temp1[0] = copy.deepcopy(t)
            lat = data.latitude.data
            long = data.longitude.data
            r2d = 45.0 / np.arctan(1.0)
            wspeed = np.sqrt(np.square(Uwind) + np.square(Vwind))
            wdir = np.arctan2(Uwind, Vwind) * r2d + 180

            nwp = dict()
            nwp['lat'] = lat
            nwp['long'] = long
            nwp['Uwind'] = Uwind
            nwp['Vwind'] = Vwind
            nwp['WS'] = wspeed
            nwp['WD'] = wdir
            nwp['Temperature'] = temp
            nwp['Cloud'] = cloud
            nwp['Flux'] = flux
            nwps[dt.strftime('%d%m%y%H%M')] = nwp

        return nwps

    def nwps_extract_for_train(self, t):
        if not os.path.exists(os.path.join(self.path_nwp, 'extract')):
            os.makedirs(os.path.join(self.path_nwp, 'extract'))
        if not os.path.exists(os.path.join(self.path_nwp, t.strftime('%Y'))):
            os.makedirs(os.path.join(self.path_nwp, t.strftime('%Y')))
        file_name1 = os.path.join(self.path_nwp, f"{t.strftime('%Y')}/Sider2_{t.strftime('%Y%m%d')}.grib")
        file_name2 = os.path.join(self.path_nwp, t.strftime('%Y') + '/SIDERT' + t.strftime('%m%d') + '00UTC.tgz')
        file_name3 = os.path.join(self.path_nwp, t.strftime('%Y') + '/H6S' + t.strftime('%m%d') + '0000/')
        nwps = dict()
        if os.path.exists(file_name1):
            nwps = self.extract_pygrib1(t, file_name1) if sys.platform == 'linux' else self.extract_cfgrib1(file_name1)
        elif os.path.exists(file_name3):
            nwps = self.extract_pygrib3(t, file_name3) if sys.platform == 'linux' else self.extract_cfgrib3(t,
                                                                                                            file_name3)
        else:
            if not os.path.exists(file_name2):
                download = DownLoader(date=t, path_nwp=self.path_nwp)
                download.download()
            if os.path.exists(file_name2):
                try:
                    nwps = self.extract_pygrib2(t, file_name2) if sys.platform == 'linux' \
                        else self.extract_cfgrib2(t, file_name2)
                except:
                    download = DownLoader(date=t, path_nwp=self.path_nwp)
                    download.download()
                    nwps = self.extract_pygrib2(t, file_name2) if sys.platform == 'linux' else self.extract_cfgrib2(t,
                                                                                                        file_name2)
        return t, nwps

    def extract_nwp(self):
        with ProcessPoolExecutor(max_workers=10) as executor:
            results = executor.map(self.nwps_extract_for_train, self.dates_ts)
            results = list(results)
        return results


