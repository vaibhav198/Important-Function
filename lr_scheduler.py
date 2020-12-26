from keras.callbacks import LearningRateScheduler

"""
Triangle learning rate scheduler means start with some lr, in initial epochs increase lr and then decrease lr.

"""

def scheduler(epoch, lr):
  if epoch <= total_epochs/2:
    if epoch%10 == 0:
      new_lr = lr*3.0
      if new_lr <= 0.01:
        return new_lr
      else:
        return lr
    else:
      return lr
  else:
    if epoch%10 == 0:
      new_lr = lr*(1/3.0)
      if new_lr >= 0.0001:
        return new_lr
      else:
        return lr
    else:
      return lr
      
lr_scheduler = LearningRateScheduler(scheduler, verbose =1)
