import tensorflow as tf
from scipy import misc
from os import listdir
from os.path import isfile, join
import data_loader
import utils
import argparse
import numpy as np
import pickle
import h5py
import time

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='train',
                       help='train/val')
	parser.add_argument('--model_path', type=str, default='Data/vgg16.tfmodel',
                       help='Pretrained VGG16 Model')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch Size')
	

	# read pretrained vgg16 network
	args = parser.parse_args()
	vgg_file = open(args.model_path,'rb')
	vgg16raw = vgg_file.read()
	vgg_file.close()

	# load the pretrained network into a tf graph
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)

	images = tf.placeholder("float", [None, 224, 224, 3])
	tf.import_graph_def(graph_def, input_map={ "images": images })

	graph = tf.get_default_graph()

	# check the loaded vgg16 network
	for opn in graph.get_operations():
		print("[VGG16] Name", opn.name, list(opn.values()))
        
	#Loading data
	all_data = data_loader.load_questions_answers()
	print(args)
	if args.split == "train":
		qa_data = all_data['training']
	else:
		qa_data = all_data['validation']
	
	image_ids = {}
	for qa in qa_data:
		image_ids[qa['image_id']] = 1

	image_id_list = [img_id for img_id in image_ids]
	print("Total Images", len(image_id_list))
	print(image_id_list[0:10])
	
	# begin extracting
	sess = tf.Session()
	idx = 0

	cnn7 = np.ndarray( (35000, 512, 49 ) )
	while idx < 35000:
		#start = time.clock()
		image_batch = np.ndarray( (args.batch_size, 224, 224, 3 ) )

		# load images into a batch
		count = 0
		for i in range(0, args.batch_size):
			if idx >= 35000:
				break
			image_file = join(args.data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]))
			# normalize image to 0~1, 224 by 224 pixels, RGB channels
			image_batch[i,:,:,:] = utils.load_image_array(image_file)[:,:,:3]
			idx += 1
			count += 1
		
		# input each batch into the vgg16 network
		feed_dict  = { images : image_batch[0:count,:,:,:] }
		# extract image feature vectors for regions, size 14 x 14 x 512 ==> 196 x 512
		cnn7_tensor = graph.get_tensor_by_name("import/pool5:0")
		cnn7_batch = sess.run(cnn7_tensor, feed_dict = feed_dict)
		cnn7_batch = np.transpose(cnn7_batch,[0,3,1,2])
		cnn7_batch = cnn7_batch.reshape(count,512,-1)
		for i in range(args.batch_size):
			cnn7_batch[i,:,:] = cnn7_batch[i,:,:] / np.linalg.norm(cnn7_batch[i,:,:],axis=0,keepdims=True)

		cnn7[(idx - count):idx, ...] = cnn7_batch[0:count, ...]
		#end = time.clock()
		print("Images Processed", idx)
		

	print("Saving cnn7 features")
	h5f_cnn7 = h5py.File( join(args.data_dir, args.split + '_cnn7.h5'), 'w')
	h5f_cnn7.create_dataset('cnn7_features', data=cnn7)
	h5f_cnn7.close()

	print("Saving image id list")
	h5f_image_id_list = h5py.File( join(args.data_dir, args.split + '_image_id_list.h5'), 'w')
	h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
	h5f_image_id_list.close()
	print("Done!")

if __name__ == '__main__':
	main()
