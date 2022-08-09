## 编译HTML版本

所有markdown文件需要在提交前清除output，它们会在服务器上重新执行生成结果。所以需要保证每个notebook执行不要太久，目前限制是20min。

在本地可以如下build html（需要GPU支持）

```
conda env update -f build/env.yml
source activate d2l-zh-build
make html
```

生成的html会在`_build/html`。

如果没有改动notebook里面源代码，所以不想执行notebook，可以使用

```
make html EVAL=0
```

但这样生成的html将不含有输出结果。

## 编译PDF版本

编译pdf版本需要xelatex、librsvg2-bin（svg图片转pdf）和思源字体。在Ubuntu可以这样安装。

```
sudo apt-get install texlive-full
sudo apt-get install librsvg2-bin
```

```
wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSansSC.zip
wget -O SourceHanSerifSC.zip https://github.com/adobe-fonts/source-han-serif/releases/download/2.001R/09_SourceHanSerifSC.zip

unzip SourceHanSansSC.zip -d SourceHanSansSC
unzip SourceHanSerifSC.zip -d SourceHanSerifSC

sudo mv SourceHanSansSC SourceHanSerifSC /usr/share/fonts/opentype/
sudo fc-cache -f -v
```


这时候可以通过 `fc-list :lang=zh` 来查看安装的中文字体。

同样的去下载和安装英文字体

```
wget -O source-serif-pro.zip https://www.fontsquirrel.com/fonts/download/source-serif-pro
unzip source-serif-pro -d source-serif-pro
sudo mv source-serif-pro /usr/share/fonts/opentype/

wget -O source-sans-pro.zip https://www.fontsquirrel.com/fonts/download/source-sans-pro
unzip source-sans-pro -d source-sans-pro
sudo mv source-sans-pro /usr/share/fonts/opentype/

wget -O source-code-pro.zip https://www.fontsquirrel.com/fonts/download/source-code-pro
unzip source-code-pro -d source-code-pro
sudo mv source-code-pro /usr/share/fonts/opentype/

sudo fc-cache -f -v
```

然后就可以编译了。

```
make pdf
```

## 其他安装

```
python -m spacy download en # 需已 pip install spacy
```

## 样式规范

贡献请遵照本教程的[样式规范](STYLE_GUIDE.md)。

## 中英文术语对照

翻译请参照[中英文术语对照](TERMINOLOGY.md)。
