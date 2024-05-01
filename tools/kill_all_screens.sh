#!/bin/bash
# 获取所有 screen 会话的名称
sessions=$(screen -ls | grep -o '[0-9]*\.[^[:space:]]*' | grep -o '[0-9]*')
# 循环遍历所有会话并强制终止它们
for session in $sessions
do
  screen -X -S $session quit
done
