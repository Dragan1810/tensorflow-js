import { load as mobilenetLoad, MobileNet } from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";
import * as knnClassifier from "@tensorflow-models/knn-classifier";

const webcamElement = document.getElementById("webcam") as HTMLVideoElement;

let net: MobileNet;

async function app() {
  console.log("Loading mobilenet..");

  // Load the model.
  net = await mobilenetLoad();
  console.log("Successfully loaded model");

  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);
  while (true) {
    const img = await webcam.capture();
    const result = await net.classify(img);

    document.getElementById("console").innerText = `
        prediction: ${result[0].className}\n
        probability: ${result[0].probability}
      `;
    // Dispose the tensor to release the memory.
    img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}

app();
