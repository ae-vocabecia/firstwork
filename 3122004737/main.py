# 论文查重
import sys
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cProfile

# 读取文件内容
def read_file(file_path):
    """
    读取指定路径的文件内容，并返回去掉空白符的字符串。
    :param file_path: 文件路径
    :return: 去掉首尾空白符后的字符串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()  # 读取文件内容并去掉首尾空白符
    except FileNotFoundError:
        # 文件未找到时，打印错误信息并退出程序
        print(f"File {file_path} not found.")
        sys.exit(1)


# 预处理文本：去除标点符号并分词
def preprocess_text(text):
    """
    预处理输入文本，去除标点符号并使用结巴分词对中文进行分词。
    :param text: 输入的原始文本
    :return: 经过标点去除和分词后的文本
    """
    # 使用正则表达式去除文本中的标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 使用结巴分词对文本进行分词，并通过空格连接分词结果
    return ' '.join(jieba.lcut(text))


# 计算相似度
def calculate_similarity(text1, text2):
    """
    计算两个文本之间的TF-IDF余弦相似度。
    :param text1: 文本1
    :param text2: 文本2
    :return: 计算得到的余弦相似度值
    """
    # 初始化TF-IDF向量化器
    vectorizer = TfidfVectorizer()
    # 将两个文本转换为TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # 计算两个文本的余弦相似度，返回第一个文本与第二个文本的相似度值
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]


# 写入相似度结果
def write_output(output_path, similarity):
    """
    将相似度结果写入指定的输出文件。
    :param output_path: 输出文件路径
    :param similarity: 计算得到的相似度值
    """
    # 以写入模式打开文件，并将相似度写入文件，保留两位小数
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{similarity:.2f}\n")


# 主函数
def main(orig_path, plagiarized_path, output_path):
    # 读取并预处理文本
    orig_text = preprocess_text(read_file(orig_path))
    plagiarized_text = preprocess_text(read_file(plagiarized_path))

    # 计算余弦相似度
    similarity = calculate_similarity(orig_text, plagiarized_text)

    # 输出相似度结果
    write_output(output_path, similarity)


if __name__ == "__main__":
    # 检查命令行参数数量
    if len(sys.argv) != 4:
        print("用法: python main.py <原文文件> <抄袭版文件> <答案文件>")
        sys.exit(1)

    # 从命令行获取路径
    orig_path = sys.argv[1]
    plagiarized_path = sys.argv[2]
    output_path = sys.argv[3]

    # 执行查重检测
    main(orig_path, plagiarized_path, output_path)
    cProfile.run("main(orig_path, plagiarized_path, output_path)", filename="performance_analysis_result")
