# 论文查重
import sys
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cProfile

# 读取文件内容
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        sys.exit(1)


# 预处理文本：去除标点符号并分词
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(jieba.lcut(text))


# 计算相似度
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]


# 写入相似度结果
def write_output(output_path, similarity):
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

