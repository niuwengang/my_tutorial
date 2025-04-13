import os
from torch.utils.data import Dataset
from typing_extensions import override


class Clip:
    def __init__(self,clip_folder_path):
        self.clip_folder_path=clip_folder_path
        self.uuid=os.path.basename(clip_folder_path)
        self.scene_type=os.path.basename(os.path.dirname(clip_folder_path))
        self.picture_list=os.listdir(os.path.join(clip_folder_path,"img"))
        print(f"ID号: {self.uuid}\n"
            f"场景类型: {self.scene_type}\n"
            f"图像帧数: {len(self.picture_list)}\n")

class MapDrDataset(Dataset):
    def __init__(self,dataset_folder_path):
        self.dataset_folder_path = dataset_folder_path
        self.clip_folder_path_list = []
        for sub_folder in os.listdir(self.dataset_folder_path):
            for clip_folder in os.listdir(os.path.join(self.dataset_folder_path,sub_folder)):
                self.clip_folder_path_list.append(os.path.join(self.dataset_folder_path,sub_folder,clip_folder))
        print(len(self.clip_folder_path_list))

    @override
    def __getitem__(self, index):
        clip=Clip(self.clip_folder_path_list[index])  
        return clip.uuid,clip.picture_list
    
    def __len__(self):
        return len(self.clip_folder_path_list)


