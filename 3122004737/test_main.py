import pytest
from unittest.mock import mock_open, patch
from main import read_file, preprocess_text, calculate_similarity, write_output, main
from sklearn.feature_extraction.text import TfidfVectorizer

# 文件操作测试
# 测试点1: 文件读取成功
def test_read_file_success():
    mock_data = "这是一个测试文件"
    with patch("builtins.open", mock_open(read_data=mock_data)) as mock_file:
        result = read_file("test.txt")
        mock_file.assert_called_once_with("test.txt", "r", encoding="utf-8")
        assert result == mock_data.strip()


# 测试点2: 文件不存在错误处理
def test_read_file_not_found():
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(SystemExit):
            read_file("non_existent.txt")


# 测试点3: 文件输出 - 正确写入相似度
def test_write_output_success():
    similarity = 0.85
    with patch("builtins.open", mock_open()) as mock_file:
        write_output("output.txt", similarity)
        mock_file.assert_called_once_with("output.txt", "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("0.85\n")


# 测试点4: 文件输出 - 文件不可写错误处理
def test_write_output_file_permission_error():
    with patch("builtins.open", side_effect=PermissionError):
        with pytest.raises(PermissionError):
            write_output("output.txt", 0.85)

# 文本预处理测试
# 测试点5: 文本预处理 - 去除标点符号
def test_preprocess_text_remove_punctuation():
    text = "这是一个测试，包含标点符号！"
    expected_output = "这是 一个 测试 包含 标点符号"
    result = preprocess_text(text)
    assert result == expected_output


# 测试点6: 文本预处理 - 分词功能
def test_preprocess_text_segmentation():
    text = "这是一个测试"
    expected_output = "这是 一个 测试"
    result = preprocess_text(text)
    assert result == expected_output


# 测试点7: 文本预处理 - 空白文本处理
def test_preprocess_empty_text():
    text = ""
    result = preprocess_text(text)
    assert result == ""


# 测试点8: 长文本处理
def test_long_text_processing():
    long_text = "这是一个非常长的测试文本。" * 10000  # 模拟非常长的文本
    result = preprocess_text(long_text)
    assert isinstance(result, str)
    assert len(result) > 0  # 检查处理后的文本不为空

# 相似度测试
# 测试点9: 计算相似度 - 完全相同文本
def test_calculate_similarity_identical_text():
    text1 = "测试文本"
    text2 = "测试文本"
    similarity = calculate_similarity(text1, text2)
    assert similarity == 1.0


# 测试点10: 计算相似度 - 完全不同文本
def test_calculate_similarity_different_text():
    text1 = "这是测试A"
    text2 = "这是测试B"
    similarity = calculate_similarity(text1, text2)
    assert similarity == 0.0


# 测试点11: 计算相似度 - 部分相似文本
def test_calculate_similarity_partial_similarity():
    text1 = "这是一个测试"
    text2 = "这是一个不同的测试"
    orig_text1 = preprocess_text(text1)
    orig_text2 = preprocess_text(text2)
    similarity = calculate_similarity(orig_text1, orig_text2)
    assert 0 < similarity < 1.0


# 测试点12: TF-IDF 向量化处理
def test_tfidf_vectorization():
    text = "我喜欢学习"
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    assert tfidf_matrix.shape == (1, len(vectorizer.get_feature_names_out()))


if __name__ == '__main__':
    pytest.main(['-vs', 'test_main.py'])
