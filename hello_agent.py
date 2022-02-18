import gym
import minerl
import logging
import logging
logging.basicConfig(level=logging.DEBUG)

env = gym.make('MineRLNavigateDense-v0')
obs = env.reset()
env.observation_space.spaces['pov']
obs['pov'].astype(np.float32) / 255.0 - 0.5
ideal_image_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)
ideal_image_space == env.observation_space.spaces['pov']
ideal_image_space.high
o = obs['pov']
frame = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
frame = np.expand_dims(frame, -1)
frame.shape


def bos(frame):
    return np.moveaxis(frame, -1, 0)


low = bos(ideal_image_space.low)
high =bos(ideal_image_space.high)
gym.spaces.Box(low=low, high=high, dtype=ideal_image_space.dtype)





n_actions = env.action_space.n
print(f'obs_space: {env.observation_space.shape[0]}, action_space: {n_actions}')
done = False
net_reward = 0
obs_array = []
count = 0
angle = [90,45,-30,-15,-90,45,0,14,28,-28,-14,0,0,0]
sum_angle = 0
while not done:
    env.render()
    action = env.action_space.noop()
    if count %2 == 0:
        action['camera'] = [45,0]
    else:
        action['camera'] = [-45,0]


    obs, reward, done, info = env.step(
        action)
    obs_array.append(obs['pov'])


    net_reward += reward
    count += 1
    # if count == 100:
    #     print(star - time.time())
    #     break
    if count > 4:
        break
action
print(time.time() - start_time)
env.observation_space.shape
obs_array[-1].shape
len(obs_array)
plt.imshow(obs_array[3])
plt.show()
for key,value in action.items():
    count += 1
    print(key)
obs['inventory']

action['craft'] = [3,3,4]
img = obs_array[0]
plt.imshow(img)
plt.show()
a = img[0][0]
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()

backSub = cv.createBackgroundSubtractorKNN()
fgMask = backSub.apply(rotated)
plt.imshow(img)
plt.show()
fgMask.shape
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
hsv_nemo = cv.cvtColor(img, cv.COLOR_RGB2HSV)
cv2.imwrite('aaa.png',img)
img = cv2.resize(img,( 256,256))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cv2
img = cv.cvtColor(img, cv.COLOR_RGB2HSV)


r, g, b = cv.split(img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")




pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

cv::Scalar(0, 140, 254), cv::Scalar(0, 165, 254)
light_orange= (255,1, 0)
dark_orange=( 255, 255,0)
from matplotlib.colors import hsv_to_rgb
lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0
plt.subplot(1, 2, 1)
plt.imshow(do_square)
plt.subplot(1, 2, 2)
plt.imshow(lo_square)
plt.show()
mask = cv.inRange(img, light_orange,dark_orange)
result = cv.bitwise_and(img, img, mask=mask)
plt.imshow(result)
plt.show()
import cv2
img = obs_array[100]
plt.imshow(img)
plt.show()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.imshow(img/255, cmap='gray')
plt.show()
ret, mask = cv2.threshold(img, 71,130,cv2.THRESH_BINARY_INV)
plt.imshow(mask, cmap='gray')
plt.show()

edge = cv2.Canny(img, 50, 250)
plt.imshow(edge,cmap='gray')
plt.show()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.show()
a = (88, 48.6 , 13.7)
b = (88,50, 25,1)
mask = cv2.inRange(hsv,b,a)
plt.imshow(obs_array[100])
plt.show()

for img in obs_array[5000:]:

    start_time = time.time()
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.resizeWindow('mask', 600, 600)
    results = model.detect([img], verbose=1)
    r = results[0]
    mask = img.astype(np.uint32).copy()
    a = visualize.draw_boxes(mask, r['rois'])
    a = np.float32(a)

    cv2.imshow("image", a)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.cla()
plt.imshow(a)
plt.show()

r = results[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'])
import os
os.path.join(data_dir, 'MINERL_DATA_ROOT')
os.environ['MINERL_DATA_ROOT']
minerl.env
data_dir='/home/dang/Documents/Doc/NISP2019/MIneRL/data/1/data_texture_0_low_res'
data = minerl.data.make('MineRLObtainDiamond-v0')
minerl.data.download('/home/dang/Documents/Doc/NISP2019/MIneRL/data/1/data_texture_0_low_res')
os.path.exists('/home/dang/Documents/Doc/NISP2019/MIneRL/data/1/MineRLObtainDiamond-v0')
# Iterate through a single epoch gathering sequences of at most 32 steps
obs
for obs, rew, done, act in data.seq_iter(num_epochs=1, max_sequence_len=32):
    print("Number of diffrent actions:", len(act))
    for action in act:
        print(act)
    print("Number of diffrent observations:", len(obs), obs)


obs['pov'].shape
for observation in obs['pov']:
    img = observation
    start_time = time.time()
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)

    cv2.imshow("image", img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

len(act['craft'])
data = minerl.data.make('MineRLTreechop-v0')
plt.imshow(obs['pov'][31])
plt.show()
image_hsv = cv2.cvtColor(obs['pov'][31], cv2.COLOR_BGR2GRAY)
cv2.imshow('img',image_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
count = 0
cv2.imwrite('./data/tree/messigray.png',img)

for obs, rew, done, act in data.seq_iter(num_epochs=1, max_sequence_len=256):
    img = obs['pov'][30]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'./data/tree/{count}.png', img)
    count += 1
    if obs['pov'].shape[0] > 120:
        img =  obs['pov'][120]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'./data/tree/{count}.png', img)
        count += 1
    if obs['pov'].shape[0] > 230:
        img =  obs['pov'][230]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'./data/tree/{count}.png',img)
        count += 1

    print("Number of diffrent actions:", len(act))
    for action in act:
        print(act)
    print("Number of diffrent observations:", len(obs), obs)
    for observation in obs:
        print(obs)
    print("Rewards:", rew)
    print("Dones:", done)


np.clip(9, -1., 1.)
env = gym.make("MsPacmanDeterministic-v4")
done = False
env.reset()
while not done:

    action = env.action_space.n
    action = [0,0,0,0,0,0,0,0,1]
    obs, reward, done, info = env.step(
        action)


