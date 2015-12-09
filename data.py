import cv2
import numpy as np
class DataProvider:
	def __init__(self, img_paths, lm_paths, rm_paths):
		self.img_paths = img_paths
		self.ll_paths = lm_paths
		self.lr_paths = rm_paths
		self.lce_imgs = []

	def __len__(self):
		return len(self.img_paths)

	def get(self, i):
		img = np.load(self.img_paths[i]).astype(np.float)
		ll_mask = cv2.imread(self.ll_paths[i])
		lr_mask = cv2.imread(self.lr_paths[i])
		lung_mask = ll_mask + lr_mask
		dsize = (512, 512)
		lung_mask = cv2.resize(lung_mask, dsize, interpolation=cv2.INTER_CUBIC)
		lung_mask = cv2.cvtColor(lung_mask, cv2.COLOR_BGR2GRAY)
		lung_mask = (lung_mask > 0).astype(np.uint8)
 		return img, lung_mask