学习连接
https://developers.google.com/codelabs/tensorflowjs-comment-spam-detection?hl=zh_cn#0
对比连接
https://glitch.com/edit/#!/comment-spam-detection-complete

然后，我们通过 package.json 和 server.js 
通过一个简单的 Node Express 服务器提供此 
www 文件夹

在tfjs-model_tutorials_spam-detection_tfjs_1文件夹中
·model.json - 这是经过训练的 TensorFlow.js 模型的一个文件。稍后，您将在 TensorFlow.js 代码中引用此特定文件。

·group1-shard1of1.bin - 这是一个二进制文件，其中包含 TensorFlow.js 模型的经过训练的权重（基本上是它用来完成分类任务的大量数字），需要将其托管在您服务器上的某个位置以供下载。

·vocab - 这种没有扩展名的奇怪文件来自 Model Maker，它展示了如何对句子中的字词进行编码，以便模型了解如何使用它们。我们将在下一部分中对此进行详细介绍。

·labels.txt - 此属性仅包含模型将预测的结果类名称。对于此模型，如果您在文本编辑器中打开此文件，其中只会列出“false”和“true”，表示“不是垃圾邮件”或“垃圾邮件”。



# .bin无法使用模型
会出现nan，nan的情况