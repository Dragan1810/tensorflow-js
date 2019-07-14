const a = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
const b = a.reshape([9, 1]);
a.print();
b.print();

a.data().then(data => console.log(data));
