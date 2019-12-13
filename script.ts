import { load as mobilenetLoad, MobileNet } from "@tensorflow-models/mobilenet";
import * as tf from "@tensorflow/tfjs";
import * as knnClassifier from "@tensorflow-models/knn-classifier";

const webcamElement = document.getElementById("webcam") as HTMLVideoElement;
const classifier = knnClassifier.create();

let net: MobileNet;

async function app() {
  console.log("Loading mobilenet..");

  // Load the model.
  net = await mobilenetLoad();
  console.log("Successfully loaded model");

  const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async (classId: string | number) => {
    console.log(classId);
    // Capture an image from the web camera.
    const img = await webcam.capture();
    const activation = net.infer(img, true);

    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
  };

  document
    .getElementById("class-a")
    .addEventListener("click", () => addExample(0));
  document
    .getElementById("class-b")
    .addEventListener("click", () => addExample(1));
  document
    .getElementById("class-c")
    .addEventListener("click", () => addExample(2));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();
      const activation = net.infer(img, true);
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ["Kamen", "Papir", "Makaze"];
      // num of trained exxamples
      console.log("NUM", classifier.getClassExampleCount());
      document.getElementById("console").innerText = `
          prediction: ${classes[result.label]}\n
          probability: ${result.confidences[result.label]}
        `;

      //  setInterval(() => {
      const gameDiv = document.getElementById("game");

      if (classes[result.label] === "Kamen") {
        gameDiv.innerHTML = `
          <img src='./images/kamen.jpg' alt='kamen' width='224' height='224' />
          `;
      }

      if (classes[result.label] === "Papir") {
        gameDiv.innerHTML = `
          <img src='./images/papir.png' alt='papir' width='224' height='224' />
          `;
      }

      if (classes[result.label] === "Makaze") {
        gameDiv.innerHTML = `
          <img src='./images/makaze.jpg' alt='makaze' width='224' height='224' />
          `;
      }
      //  }, 100);

      img.dispose();
    }

    await tf.nextFrame();
  }
}

app();
