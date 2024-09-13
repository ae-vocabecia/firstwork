import os
import pytest
from main import read_file, preprocess_text, calculate_similarity, write_output


# 创建临时文件的 fixture
@pytest.fixture
def temp_file(tmpdir):
    def _create_temp_file(filename, content):
        file_path = tmpdir.join(filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(file_path)

    return _create_temp_file


# 2.1. 文件读取功能测试
def test_read_file(temp_file):
    test_content = "这是一个测试。"
    file_path = temp_file('test_orig.txt', test_content)
    result = read_file(file_path)
    assert result == test_content


# 2.2. 文本预处理功能测试
def test_preprocess_text():
    test_text = "这是一个测试。"
    result = preprocess_text(test_text)
    expected = "这是 一个 测试"
    assert result == expected


# 2.3. 余弦相似度计算功能测试
def test_calculate_similarity():
    text1 = "这是一个测试"
    text2 = "这是一个简单的测试"
    text1=preprocess_text(text1)
    text2=preprocess_text(text2)
    result = calculate_similarity(text1, text2)
    assert 0.7 < result < 1.0  # 相似度应接近 0.8


# 2.4. 输出相似度到文件测试
def test_write_output(temp_file):
    file_path = temp_file('output.txt', "")
    similarity = 0.85
    write_output(file_path, similarity)

    with open(file_path, 'r', encoding='utf-8') as f:
        result = f.read().strip()

    assert result == "0.85"


# 2.5. 主函数执行流程测试
def test_main_process(temp_file, monkeypatch):
    # 模拟两个输入文本文件和输出文件路径
    orig_file_path = temp_file('orig.txt', '这是原始文本。')
    plagiarized_file_path = temp_file('plagiarized.txt', '这是抄袭文本。')
    output_file_path = temp_file('output.txt', '')

    # 模拟命令行参数
    monkeypatch.setattr('sys.argv', ['', orig_file_path, plagiarized_file_path, output_file_path])

    # 导入并执行主程序
    from main import main
    main(orig_file_path, plagiarized_file_path, output_file_path)

    # 验证输出
    with open(output_file_path, 'r', encoding='utf-8') as f:
        result = f.read().strip()
    assert 0.0 <= float(result) <= 1.0




if __name__ == '__main__':
    pytest.main(['-vs', 'test_similarity.py'])