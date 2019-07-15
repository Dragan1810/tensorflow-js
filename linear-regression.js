//canvas

const xs = [];
const ys = [];

let m, b;

const leariningRate = 0.5;
const optimizer = tf.train.sgd(leariningRate);

function setup() {
  createCanvas(400, 400);

  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels) {
  return pred
    .sub(labels)
    .square()
    .mean();
}

function predict(xs) {
  const tfxs = tf.tensor1d(xs);
  // y = mx + b;

  const ys = tfxs.mul(m).add(b);
  return ys;
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);

  xs.push(x);
  ys.push(y);
}

function draw() {
  if (xs.length > 0) {
    const tfys = tf.tensor1d(ys);
    optimizer.minimize(() => loss(predict(xs), tfys));
  }
  background(0);

  stroke(255);
  strokeWeight(4);

  for (let i = 0; i < xs.length; i++) {
    let px = map(xs[i], 0, 1, 0, width);
    let py = map(ys[i], 0, 1, height, 0);

    point(px, py);
  }

  const XS = [0, 1];
  const yss = tf.tidy(() => predict(XS));

  let x1 = map(XS[0], 0, 1, 0, width);
  let x2 = map(XS[1], 0, 1, 0, width);

  let liney = yss.dataSync();
  yss.dispose();
  let y1 = map(liney[0], 0, 1, height, 0);
  let y2 = map(liney[1], 0, 1, height, 0);

  line(x1, y1, x2, y2);
}
