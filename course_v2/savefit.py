import os

import tensorflow as tf

asis=tf.autograph.experimental.do_not_convert

import inspect
def savefile():
  callerframerecord = inspect.stack()[2]
  frame = callerframerecord[0]
  info = inspect.getframeinfo(frame)
  fn = info.filename
  bn = os.path.splitext(os.path.basename(info.filename))[0]
  if info.filename.endswith('.py'):
    return '../data/h5'+bn+'%s.h5'%info.lineno
  else:
    return '../data/h5'+bn.strip('>').strip('<')+'%s.h5'%info.lineno

def savefit(m,*a,**ka):
  saved = savefile()
  if os.path.exists(saved):
    m = tf.keras.models.load_model(saved)
    history = None
  else:
    history = m.fit(*a,**ka)
    m.save(saved)
  return m,history
