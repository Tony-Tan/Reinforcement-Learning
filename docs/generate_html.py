import json
import os
from shutil import copyfile
import re
import jinja2

def generate_home(algorithm_json_path, output_html_path):
    # Parse the Python script
    section = {'id': f'section-0', 'title': '', 'description': ''}
    section['title'] = 'HOME'
    with open(algorithm_json_path, 'r') as file:
        data = json.load(file)
        for cat in data["algorithms"]:
            section['description'] += f'## {cat}\n\n'
            i = 0
            for algo in data["algorithms"][cat]:
                algo_name = algo["name"]
                file_name = algo_name.replace(' ','_')
                section['description'] += f'{i}. [{algo_name}](./{file_name}.html)\n\n'
                i += 1
    # Setup Jinja2 environment
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
    template = env.get_template('home_temp.html')

    # Render the template with the sections data
    html_output = template.render(section=section)

    # Write the output to an HTML file
    with open(output_html_path, 'w') as file:
        file.write(html_output)

    print(output_html_path + " HOME page generated successfully!")


def generate_page(script_path, output_html_path):
    # Parse the Python script
    sections = parse_python_script(script_path)
    # Setup Jinja2 environment
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
    template = env.get_template('page_temp.html')

    # Render the template with the sections data
    html_output = template.render(sections=sections)

    # Write the output to an HTML file
    with open(output_html_path, 'w') as file:
        file.write(html_output)

    print(script_path + " HTML page generated successfully!")


def replace_multiline_strings(lines):
    text = "\n".join(lines)  # 将行列表转换为单个字符串以便处理
    # 使用正则表达式匹配 ''' 或 """ 之间的所有内容，包括多行
    # re.DOTALL 使得 . 匹配包括换行符在内的所有字符
    text = re.sub(r"'''(.*?)'''|\"\"\"(.*?)\"\"\"", '', text, flags=re.DOTALL)
    return text.split('\n\n')  # 将处理后的字符串再次分割为行列表


def parse_python_script(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 处理多行字符串，去除多行字符串内容
    lines = replace_multiline_strings(lines)
    # 保存每个部分的数据
    sections = []
    current_section = None
    code_block = []
    section_id = 1

    is_in_comment_block = False
    comment_lines = []
    collecting_code = False  # 用于指示是否开始收集代码行

    for line_num, line in enumerate(lines, start=1):
        line_stripped = line.strip().strip('\n')
        if line_stripped.startswith('#'):
            # 检测到注释行，开始或继续注释块
            if current_section and collecting_code:
                # Remove trailing empty lines from code block
                while code_block and code_block[-1]['content'].strip() == '':
                    code_block.pop()
                current_section['code'] = code_block
                sections.append(current_section)
                section_id += 1
                code_block = []

            # 创建新的 section
            current_section = {'id': f'section-{section_id}', 'title': '', 'description': '', 'lang': 'python',
                               'code': []}

            is_in_comment_block = True
            collecting_code = False  # 结束代码收集

            # 提取注释内容，考虑空注释行
            comment_line = repr(line_stripped[1:].strip())[1:-1] if len(line_stripped) > 1 else '\n'
            comment_lines.append(comment_line)

        else:
            if is_in_comment_block:
                # 处理收集的注释块
                full_comment = ''.join(comment_lines)
                if comment_lines[0].startswith('#'):
                    title_level = len(re.match(r'^#+', comment_lines[0]).group(0))
                    current_section['title'] = comment_lines[0][title_level:].strip()
                current_section['description'] = full_comment
                comment_lines = []
                is_in_comment_block = False

            if line_stripped == '' and not collecting_code:
                # 忽略代码块前的空行
                continue

            # 开始收集代码行
            collecting_code = True
            code_block.append({'line_num': str(line_num), 'content': line.rstrip()})

    # 检查是否有未处理的注释块
    if is_in_comment_block and comment_lines:
        full_comment = ''.join(comment_lines)
        if comment_lines[0].startswith('#'):
            title_level = len(re.match(r'^#+', comment_lines[0]).group(0))
            current_section['title'] = comment_lines[0][title_level:].strip()
        current_section['description'] = full_comment

    # 保存最后一个代码块
    if current_section or code_block:
        if not current_section:
            current_section = {'id': f'section-{section_id}', 'title': '', 'description': '', 'lang': 'python',
                               'code': []}
        current_section['code'] = code_block
        sections.append(current_section)
    return sections


def read_json(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)


def doc_website(json_file, output_dir):
    data = read_json(json_file)
    # 生成主页
    generate_home(json_file, os.path.join(output_dir, "index.html"))
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for cat in data["algorithms"]:
        for algo in data["algorithms"][cat]:
            algo_name = algo["name"]
            script_path = algo["script_path"]
            file_name = algo_name.replace(' ', '_')
            output_html_path = os.path.join(output_dir, f"{file_name}.html")

            # 生成每个算法的HTML页面
            generate_page(script_path, output_html_path)


# 调用函数
if __name__ == '__main__':
    doc_website('.algs_path.json', 'htmls/')
