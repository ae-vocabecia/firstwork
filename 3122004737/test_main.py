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



if __name__ == '__main__':
    pytest.main(['-vs', 'test_main.py'])