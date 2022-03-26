import numpy as np
from keras.models import load_model
from keras.preprocessing import image


class good_bad:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        try:
            model = load_model('model.h5')
            print("model loaded")

            imagename = self.filename
            test_image = image.load_img(imagename, target_size=(247, 326))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)

            if int(result[0][0]) == 0:
                prediction = 'You seems to be good human. Are you?'
                return [{"image": prediction}]
            else:
                prediction = 'My model says that you are a bad human. I am not saying that.'
                return [{"image": prediction}]
        except Exception as e:
            print(e)
