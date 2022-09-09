./app目录下，存放着处理文本的模块，preprocess包含文本预处理（去符号去停用词tokenization等），statistic模块包含文本统计（词频、共现词、词云等），model模块包含用LDA作主题推断，text alignment模块包含句子相似度计算

./data目录下是文本数据，作测试用，可以在notebook中找到对应用法

./notebook下是调用./app中的函数进行文本处理的例子，包含一些情况说明

./pipeline是把./app中的函数组装好的pipeline，可以直接输入文本文件使用

./process目录下是最开始的一些只实现了部分功能的函数和用来测试的notebook
