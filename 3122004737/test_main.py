import pytest
from unittest.mock import mock_open, patch
from main import read_file, preprocess_text, calculate_similarity, write_output, main
from sklearn.feature_extraction.text import TfidfVectorizer

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

# 测试点8: 特殊字符文本的处理
def test_preprocess_text_special_characters():
    text = "你好😊 这是测试"
    expected_output = "你好 这是 测试"
    result = preprocess_text(text)
    assert result == expected_output

# 测试点9: 长文本处理
def test_long_text_processing():
    long_text = "这是一个非常长的测试文本。" * 10000  # 模拟非常长的文本
    result = preprocess_text(long_text)
    assert isinstance(result, str)
    assert len(result) > 0  # 检查处理后的文本不为空

if __name__ == '__main__':
    pytest.main(['-vs', 'test_main.py'])