import atexit
import functools
import sys
import threading
import traceback
import gym
import numpy as np
import skimage.transform
from PIL import Image
import matplotlib.pyplot as plt
class DeepMindControl:

  def __init__(self, name, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True):
    import gym
    version = 0 if sticky_actions else 4
    #name = ''.join(word.title() for word in name.split('_'))
#    with self.LOCK:
#      self._env = gym.make('{}NoFrameskip-v{}'.format(name, version))
    with self.LOCK:
      self._env = gym.make(name,full_action_space=True)
    self._name = name
    print("succes making environment!")
    self._action_repeat = action_repeat
    self._size = size
    self._grayscale = grayscale
    self._noops = noops
    self._life_done = life_done
    self._lives = None
    shape = self._env.observation_space.shape[:2] + (() if grayscale else (3,))
    self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(2)]
    self._random = np.random.RandomState(seed=None)

  @property
  def observation_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      self._env.reset()
    noops = self._random.randint(1, self._noops + 1)
    for _ in range(noops):
      done = self._env.step(0)[2]
      if done:
        with self.LOCK:
          self._env.reset()
    self._lives = self._env.ale.lives()
    if self._grayscale:
      self._env.ale.getScreenGrayscale(self._buffers[0])
    else:
      self._env.ale.getScreenRGB2(self._buffers[0])
    self._buffers[1].fill(0)
    return self._get_obs()

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      _, reward, done, info = self._env.step(action)
      total_reward += reward
      if self._life_done:
        lives = self._env.ale.lives()
        done = done or lives < self._lives
        self._lives = lives
      if done:
        break
      elif step >= self._action_repeat - 2:
        index = step - (self._action_repeat - 2)
        if self._grayscale:
          self._env.ale.getScreenGrayscale(self._buffers[index])
        else:
          self._env.ale.getScreenRGB2(self._buffers[index])
    obs = self._get_obs()
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode)

  def _get_obs(self):
    if self._action_repeat > 1:
      np.maximum(self._buffers[0], self._buffers[1], out=self._buffers[0])
    image = np.array(Image.fromarray(self._buffers[0]).resize(self._size, Image.BILINEAR))
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = image[:, :, None] if self._grayscale else image
    return {'image': image}

class ObservationDict(object):

  def __init__(self, env, key='observ'):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self._env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = {self._key: np.array(obs)}
    return obs


class ObservationToRender(object):

  def __init__(self, env, key='image'):
    self._env = env
    self._key = key
    self._image = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    return gym.spaces.Dict({})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._image = obs.pop(self._key)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    self._image = obs.pop(self._key)
    return obs

  def render(self, *args, **kwargs):
    return self._image

class PixelObservations(object):

  def __init__(
      self, env, size=(64, 64), dtype=np.uint8, key='image',
      render_mode='rgb_array', name=None, p = None, robot=None):
    assert isinstance(env.observation_space, gym.spaces.Dict)            
    self._env = env
    self._p = p
    self._name = name
    self._size = size
    self._dtype = dtype
    self._key = key
    self._render_mode = render_mode
    self._robot = robot    

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    high = {np.uint8: 255, np.float: 1.0}[self._dtype]
    image = gym.spaces.Box(0, high, self._size + (3,), dtype=self._dtype)
    spaces = self._env.observation_space.spaces.copy()
    assert self._key not in spaces
    spaces[self._key] = image
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs[self._key] = self._render_image()
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
   # self._p.resetSimulation()
   # self._p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
   # self._p.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    obs[self._key] = self._render_image()
    return obs
    
  def _render_image(self):
    #print("bodies: " + str(self._p.getNumBodies()))
    #camTargetPos =[0,0,5]# location of the target (focus) of the camera
    #print(self._robot.bodyxyz)
    #print(self._p.getBasePositionAndOrientation(0))    
    #print(self._p.getBasePositionAndOrientation(1))
   # print(self._p.getBasePositionAndOrientation(1)[0])
  #  print(self._p.getNumJoints(0))
   # print(self._p.getNumJoints(1))
                        
    #camTargetPos, _ = self._p.getBasePositionAndOrientation(1)# location of the target (focus) of the camera
    if "Pendulum" not in self._env._name and "Reacher" not in self._env._name:
      camTargetPos = self._p.getLinkState(1, 0, computeForwardKinematics=True)[4]
    else:
      camTargetPos, _ = self._p.getBasePositionAndOrientation(1)# location of the target (focus) of the camera
    #print(camTargetPos)
    #print(self._p.getBasePositionAndOrientation(1)[0])
    if "Reacher" in self._env._name:
      camDistance = 1.05 # tilt the camera relative to the target
    elif "Thrower" in self._env._name or "Pusher" in self._env._name:
      camDistance = 1.2
    else:
      camDistance = 2
    yaw = 0 # yaw angle relative to the target
    if "Reacher" in self._env._name:
      pitch = 90 # tilt the camera relative to the target
    elif "Pusher" in self._env._name:
      pitch = 45
    else:
      pitch = 0
    roll = 0 # the camera roll angle relative to the target
    upAxisIndex = 2 # vertical axis of the camera (z)
    fov = 60 # camera angle of view
    nearPlane = 1 # distance to the nearest clipping plane
    farPlane = 25 # distance to the far clipping plane
    pixelWidth = 64 # width of the image
    pixelHeight = 64 # the height of the image is
    aspect = pixelWidth /pixelHeight; # aspect ratio of the image
    # The view matrix is ??
    viewMatrix = self._p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
    projectionMatrix = self._p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
    #Rendering of the image from the camera
    img_arr = self._p.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow = 1, lightDirection =[0,1,1], renderer = self._p.ER_TINY_RENDERER)
    w = img_arr[0]#width of the image, in pixels
    h = img_arr[1]#height of the image, in pixels
    rgb = img_arr[2]#color data RGB
    dep = img_arr[3]#depth data
    rgba = Image.fromarray(rgb)
    #plt.imsave('./test1.jpg', np.asarray(rgba).astype(np.uint8))
    rgb_img = Image.new("RGB", (w,h), (255, 255, 255))
    #plt.imsave('./test2.jpg', np.asarray(rgb_img).astype(np.uint8))
    rgb_img.paste(rgba, mask=rgba.split()[3]) # 3 is the alpha channel
    #plt.imsave('./test3.jpg', np.asarray(rgb_img).astype(np.uint8))
    image = np.clip(rgb_img, 0, 255).astype(np.uint8)
   # plt.imsave('./test4.jpg', image)
    return image

class ConvertTo32Bit(object):

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    observ, reward, done, info = self._env.step(action)
    #observ = self.map_(self._convert_observ, observ)
    reward = self._convert_reward(reward)
    return observ, reward, done, info

  def reset(self):
    observ = self._env.reset()
    #observ = self.map_(self._convert_observ, observ)
    return observ

  def _convert_observ(self, observ):
    if not np.isfinite(observ).all():
      raise ValueError('Infinite observation encountered.')
    if observ.dtype == np.float64:
      return observ.astype(np.float32)
    if observ.dtype == np.int64:
      return observ.astype(np.int32)
    return observ

  def _convert_reward(self, reward):
    if not np.isfinite(reward).all():
      raise ValueError('Infinite reward encountered.')
    return np.array(reward, dtype=np.float32)
    
  def map_(function, *structures, **kwargs):
      # Named keyword arguments are not allowed after *args in Python 2.
      flatten = kwargs.pop('flatten', False)
      assert not kwargs, 'map() got unexpected keyword arguments.'



class PyBullet:
  def __init__(self, env, name, size=(64, 64), camera=None, action_repeat = 4, render=False):
    
    self._name = name
    self._grayscale = False
    self._env = env
    self._size = size
    self._action_repeat = action_repeat
    #if camera is None:
    #  camera = dict(quadruped=2).get(domain, 0)
    self._camera = env.camera
    shape = self._env.observation_space.shape[:2] + (() if self._grayscale else (3,))
    self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(2)]
    
    
  @property
  def observation_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def step(self, action):
    _ , reward, done, info = self._env.step(action)
    print(self._env.p.getLinkState(1, 0, computeForwardKinematics=True)[4])
    obs = self.render('rgb_array')
    image = np.array(Image.fromarray(obs).resize(
        self._size, Image.BILINEAR))
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = image[:, :, None] if self._grayscale else image
    obs = {'image': image}
    return obs, reward, done, info

  def reset(self):
    self._env.reset()
    obs = self.render('rgb_array')
    
    image = np.array(Image.fromarray(obs).resize(
        self._size, Image.BILINEAR))
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = image[:, :, None] if self._grayscale else image
    return {'image': image}


  def render(self, mode):
    return self._env.render(mode)


class Collect:

  def __init__(self, env, name, callbacks=None, precision=32):
    self._env = env
    self._name = name
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['task'] = self._name
    transition['reward'] = reward
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.shape)
    transition['reward'] = 0.0
    transition['task'] = self._name
    transition['discount'] = 1.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    elif isinstance(value[0], str):
      dtype = np.string_
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      obs, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return obs, total_reward, done, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class ObsDict:

  def __init__(self, env, key='obs'):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self._env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = {self._key: np.array(obs)}
    return obs


class OneHotAction:

  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    if not np.allclose(reference, action):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference

class PadActions(object):
  """Pad action space to the largest action space."""

  def __init__(self, env, spaces, name):
    self._name = name
    self._env = env
    self._action_space = self._pad_box_space(spaces)

  @property
  def observation_space(self):
    return self._env.observation_space

  @property
  def action_space(self):
    return self._action_space

  def step(self, action, *args, **kwargs):
    action = action[:len(self._env.action_space.low)]
    return self._env.step(action, *args, **kwargs)

  def reset(self, *args, **kwargs):
    return self._env.reset(*args, **kwargs)

  def _pad_box_space(self, spaces):
    assert all(len(space.low.shape) == 1 for space in spaces)
    length = max(len(space.low) for space in spaces)
    low, high = np.inf * np.ones(length), -np.inf * np.ones(length)
    for space in spaces:
      low[:len(space.low)] = np.minimum(space.low, low[:len(space.low)])
      high[:len(space.high)] = np.maximum(space.high, high[:len(space.high)])
    return gym.spaces.Box(low, high, dtype=np.float32)


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs


class Async:

  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, ctor, strategy='process'):
    self._strategy = strategy
    if strategy == 'none':
      self._env = ctor()
    elif strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      import multiprocessing as mp
    else:
      raise NotImplementedError(strategy)
    if strategy != 'none':
      self._conn, conn = mp.Pipe()
      self._process = mp.Process(target=self._worker, args=(ctor, conn))
      atexit.register(self.close)
      self._process.start()
    self._obs_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._obs_space:
      self._obs_space = self.__getattr__('observation_space')
    return self._obs_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    if self._strategy == 'none':
      return getattr(self._env, name)
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    blocking = kwargs.pop('blocking', True)
    if self._strategy == 'none':
      return functools.partial(getattr(self._env, name), *args, **kwargs)
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    promise = self._receive
    return promise() if blocking else promise

  def close(self):
    if self._strategy == 'none':
      try:
        self._env.close()
      except AttributeError:
        pass
      return
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    return self.call('step', action, blocking=blocking)

  def reset(self, blocking=True):
    return self.call('reset', blocking=blocking)

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except ConnectionResetError:
      raise RuntimeError('Environment worker crashed.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError(f'Received message of unexpected type {message}')

  def _worker(self, ctor, conn):
    try:
      env = ctor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          break
        raise KeyError(f'Received message of unknown type {message}')
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error in environment process: {stacktrace}')
      conn.send((self._EXCEPTION, stacktrace))
    conn.close()
