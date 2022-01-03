dir_egg="./torch_frame.egg-info"
dir_dist="./dist"
dir_build="./build"

[ -d "$dir_dist" ] && rm -rf "$dir_dist" && echo "清理${dir_egg}缓存成功"

python setup.py sdist bdist_wheel

[ -d "$dir_build" ] && rm -rf "$dir_build" && echo "清理${dir_egg}缓存成功"
[ -d "$dir_egg" ] && rm -rf "$dir_egg" && echo "清理${dir_egg}缓存成功"

echo "done"


