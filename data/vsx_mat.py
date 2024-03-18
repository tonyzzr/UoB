from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np


@dataclass
class MatData:
    pdata: dict
    imgdata: dict
    trans: dict

    def __str__(self) -> str:
        return f'pdata: {len(self.pdata.keys())} x {self.pdata[0].keys()} \n' + \
               f'imgdata: {len(self.imgdata.keys())} x {self.imgdata[0].shape} \n' + \
               f'trans: {self.trans.keys()}'

class MatDataLoader:
    def __init__(self, path:Path) -> None:
        self.data = h5py.File(str(path))

    def build_mat_data(self) -> MatData:
        return MatData(
            pdata    = self.__load_pdata(),
            imgdata = self.__load_imgdata(),
            trans   = self.__load_trans(),
        )

    def __load_pdata(self) -> dict:
        '''
            Load the PData field of the matFile.
        '''
        import numpy as np

        PData = {}
        PDataRef = self.data['PData']


        PDataNum = len(list(PDataRef['PDelta']))
        # print(PDataNum)

        for i in range(PDataNum):
            PDeltaRef = list(PDataRef['PDelta'])[i]
            SizeRef   = list(PDataRef['Size'])[i]
            OriginRef = list(PDataRef['Origin'])[i]

            PDelta = np.array(self.data[PDeltaRef[0]]).T
            Size = np.array(self.data[SizeRef[0]]).T
            Origin = np.array(self.data[OriginRef[0]]).T

            PDataDict = {
                'idx'    : i,

                'PDelta' : PDelta[0],
                'Size'   : Size[0],
                'Origin' : Origin[0],
                
                'dx_wl'  : PDelta[0][0],
                'dz_wl'  : PDelta[0][2],
                'Ox_wl'  : Origin[0][0],  # x-origin of the PData (ref. Trans.ElePos)
                'Oz_wl'  : Origin[0][2],  # z-origin of the PData (ref. Trans.ElePos)

                'nx'     : Size[0][1],
                'nz'     : Size[0][0],
            }

            PData[i] = PDataDict
        
        # print(PData)
        return PData

    def __load_imgdata(self) -> dict:
        '''
            Load the ImgData field of the matFile.
        '''
        # Note: check how many cells in the ImgData field, return respectively.

        ImgData = {}
        ImgDataVar = self.data['ImgData'][0]
        nField, = ImgDataVar.shape
        # print(nField)
        for i in range(nField):
            ref = ImgDataVar[i]
            data = np.array(self.data[ref])
            ImgData[i] = data


        return ImgData

    def __load_trans(self) -> dict:
        '''
            Load the Trans field of the matFile.
        '''
        Trans = {}

        TransVar = self.data['Trans']
        TransKeys = list(TransVar)

        for key in TransKeys:
            value = np.array(TransVar[key]).T
            r, c = value.shape
            if (r == 1) or (c == 1):
                value = np.reshape(value, (r*c, ),)
                r_, = value.shape
                if r_ == 1:
                    value = value[0]
            
            # ---> Special treat for 'name' and 'units'
            Trans[key] = np.array(value)
            
            # print(key, value)
            # input()
        
        Trans['SoundSpeed'] = 1540 # (m/s)
        Trans['wavelengthMm'] = Trans['SoundSpeed'] / (Trans['frequency'] * 1e3)

        # print(Trans)
        # input()

        
        return Trans


'''
    Load mat_data from a given directory.
'''
import glob
import os

class DirMatDataLoader:
    def __init__(self, matfile_dir) -> None:

        self.matfile_dir = matfile_dir
        self.file_index = self.get_file_index(self.matfile_dir)

    def load_matfile(self, ind):

        matfile_paths = {
            'lftx': fr'{self.matfile_dir}\{ind}_LF.mat',
            'hftx': fr'{self.matfile_dir}\{ind}_HF.mat',
        }

        mat_data = {}

        for key in matfile_paths:
            mat_data[key] = MatDataLoader(matfile_paths[key]).build_mat_data()
            
        return mat_data

    def load_all_matfiles(self, ):

        mat_data = {}
        for ind in range(1, self.file_index+1):
            single_file_mat_data = self.load_matfile(ind)

            if ind == 1:
                mat_data = single_file_mat_data

            else:
                for key in single_file_mat_data:
                    mat_data[key].imgdata = self.concatenate_imgdata(prev = mat_data[key].imgdata,
                                                                     new =single_file_mat_data[key].imgdata)

        return mat_data

    @staticmethod
    def concatenate_imgdata(prev, new):

        concatenated = {}
        for view_ind in prev.keys():
            concatenated[view_ind] = np.concatenate(
                (prev[view_ind], new[view_ind]),
                axis = 0,
            )
         
        return concatenated

    @staticmethod
    def get_file_index(matfile_dir):
        '''
            1. get all .mat filenames
            2. check number of files -- should be an even number -- and divide it by 2 (assumptions here, update if necessary)
        '''

        pattern = os.path.join(matfile_dir, '*.mat')
        mat_files = glob.glob(pattern)

        if len(mat_files) % 2 == 0:
            return len(mat_files) // 2
        else:
            print(mat_files)
            raise ValueError("number of matfiles is not even")





def main():
    return

if __name__ == "__main__":
    main()
