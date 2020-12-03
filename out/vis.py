import pickle

resul_pkl = "/data/amax/b510/ly/mmdetection3d/out/result.pkl"
store_path = '/data/amax/b510/ly/mmdetection3d/out/hardvoxelpred_2'
with open(resul_pkl, 'rb') as ff:
	inf = pickle.load(ff)
	for ano in inf:
		name = ano['name']
		truncated = ano['truncated']
		occluded = ano['occluded']
		alpha = ano['alpha']
		bbox = ano['bbox']
		dimensions = ano['dimensions'] #lwh
		location = ano['location']
		rotation_y = ano['rotation_y']
		score = ano['score']
		boxes_lidar = ano['boxes_lidar']
		frame_id = ano['frame_id']
		store_file = store_path + ('%s.txt' % frame_id)
		with open(store_file, 'a') as f:
			for i in range(len(name)):
				line = '%s %f %d %f %f %f %f %f %f %f %f %f %f %f %f' % (name[i], truncated[i], occluded[i], alpha[i], bbox[i][0], \
					bbox[i][1], bbox[i][2], bbox[i][3], dimensions[i][1], dimensions[i][2], dimensions[i][0],  \
					location[i][0], location[i][1], location[i][2], rotation_y[i])		
				print('writing frame :' , frame_id)			
				f.write(line)
				f.write('\n')

		f.close()