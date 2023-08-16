import logging
import glob

import torch

class FolderData(torch.utils.data.Dataset):
    def __init__(self, path, transforms, extensions=['.jpg', '.png'], recursive=False, verbose=False):
        self.verbose = verbose
        if self.verbose:
            logger = logging.getLogger(__name__)
        
        if len(extensions) == 0:
            if self.verbose:
                logger.error("Expected at list one extension, but none was received.")
            raise ValueError

        if self.verbose:
            logger.info("Constructing the list of images.")
        additional_pattern = '/**/*' if recursive else '/*'
        files = []
        for extension in extensions:
            files.extend(glob.glob(path + additional_pattern + extension, recursive=recursive))

        if self.verbose:
            logger.info("Finished searching for images. %s images found", len(files))
            logger.info("Preparing to run the detection.")
        
        self.files = files
        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = self.transforms(image_path)

        return image_path, image

    def __len__(self):
        return len(self.files)