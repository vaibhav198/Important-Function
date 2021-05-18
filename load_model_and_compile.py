#Suppose, we want to load a pretrained model and want to resume training----> to do this we have two ways
#1st way ----> initialize the model and then load the weights (But if do not have separate weights, 
#means you have saved model weights with architecture)
#2nd way is, we have save model weights and architecture both, so we can load the model and resume the training easily, but what if we'll
# face errors in resuming the training. The one option is, we can compile it but that's not a right way 
#because as soon as we compile it our loss and optimzer state will initialize. So in this situation the following two lines of code will help.

#model = load_model('IOU_model_fold0_aug_8may.h5', custom_objects = {'IOU': IOU})
model = load_model("IOU_model_fold0_aug_8may.h5", custom_objects={"IOU":IOU}, compile=True)
model.compile(loss=model.loss, optimizer=model.optimizer, metrics=[IOU])
