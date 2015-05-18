@echo off
for /r %%i in (left\*) do @echo L %%i >> image_list.txt
for /r %%i in (right\*) do @echo R %%i >> image_list.txt
pause