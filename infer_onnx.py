import argparse

import cv2
import numpy as np
import onnxruntime as ort


class PedestrianONNX:
    def __init__(self, model_path,
                 height=256,
                 width=192,
                 threshold=0.5) -> None:
        self.ort_sess = ort.InferenceSession(model_path)
        self.height = height
        self.width = width
        self.threshold = threshold

    def preprocess(self, img_path,
                   scale = 1/255,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = np.array(img)
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32)
        img = img * scale
    
        shape = (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')
    
        img = img - self.mean
        img = img / self.std
        batch = np.expand_dims(img, axis=0).astype(np.float32)
        batch = np.transpose(batch, (0, 3, 1, 2))
        return batch

    def forward(self, input):
        outputs = self.ort_sess.run(None, {'input': input})
        return outputs

    def decode(self,
               output,
               threshold=0.5,
               label_list=[
                    'Hat', 'Glasses', 'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo',
                    'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern', 'LongCoat',
                    'Trousers', 'Shorts', 'Skirt&Dress', 'boots', 'HandBag', 'ShoulderBag',
                    'Backpack', 'HoldObjectsInFront', 'AgeOver60', 'Age18-60', 'AgeLess18',
                    'Female', 'Front', 'Side', 'Back'
               ]):
        print(output)
        output = output[0][0]
        label = np.array(label_list)[output > threshold]
        score = output[output > threshold]
        pred = np.array(list(zip(label, score)))
        return pred

    def predict(self, img):
        input = self.preprocess(img)
        output = self.forward(input)
        pred = self.decode(output, self.threshold)
        return pred


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--img_path", type=str, required = True)
    parser.add_argument("--model_path", type=str, required = True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    model = PedestrianONNX(args.model_path, threshold=args.threshold)
    pred = model.predict(args.img_path)
    print(pred)
