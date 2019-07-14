const model = tf.sequential();

const config_hidden = {
  units: 4,
  inputShape: [2],
  activation: 'sigmoid'
};

const config_output = {
  units: 3,
  activation: 'sigmoid'
};

const hidden = tf.layers.dense(config_hidden);
const output = tf.layers.dense(config_output);

model.add(hidden);
model.add(output);

const optimizer = tf.train.sgd(0.1);

const config = {
  optimizer,
  loss: 'meanSquaredError'
};

model.compile(config);

const xs = tf.tensor2d([
  [0.25, 0.92],
  [0.35, 0.02],
  [0.45, 0.12],
  [0.55, 0.72]
]);
let ys = model.predict(xs);
ys.print();
