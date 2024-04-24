Команда для установки среды test_3
```bash
    conda env create -f environment.yml
```
Команда для обновления существующей среды test_3, если нужно
```bash
    conda env update --file environment.yml  --prune
```
Команда для активации среды test_3
```bash
    conda activate test_3
```
Главный файл main.py

Самая медленная часть кода - это функция tools.pm_model(), поэтому я туда подаю частичные данные (Баесовский подход). Её вроде можно ускорить с помощью gpu, но это не точно (https://jax.readthedocs.io/en/latest/installation.html, https://www.pymc.io/projects/examples/en/latest/howto/wrapping_jax_function.html) и это доступно только на linux
Выходные данные представлены в папке output.
