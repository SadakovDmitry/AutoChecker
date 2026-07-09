# Auto Classifier Validator

`auto_classifier` - это гибридный валидатор разметки для чатов поддержки.

Он **не заменяет текущий классификатор**. Его задача другая: проверить уже
готовую пару `чат + причина, которую выбрал классификатор` и решить, можно ли
эту разметку принять автоматически.

На выходе для каждой строки программа считает:

- `p_correct` - вероятность, что причина классификатора корректна;
- `decision` - `accept` или `review`;
- `threshold` - порог авто-приемки для этой причины;
- `p_lsa`, `p_embedding` - отдельные сигналы моделей;
- ближайшие похожие правильные и неправильные исторические примеры;
- `rule_flags` - простые признаки вроде `operator_only_keywords` или
  `no_client_text`.

Главная идея первой версии: **лучше принять меньше чатов, но с высокой
точностью**, чем автоматически пропустить много ошибок. Все спорные случаи
уходят в `review`, размечаются руками и потом добавляются в обучение.

## Когда это использовать

Подходит для сценария, где уже есть классификатор, который размечает чаты по
тематикам, а аналитик потом вручную проверяет `да/нет`.

Вместо полной ручной проверки схема становится такой:

```text
новый чат
  -> текущий классификатор
  -> reason_id
  -> auto_classifier validator
  -> p_correct + accept/review
```

Например:

```text
reason_id=5, p_correct=0.97, decision=accept
```

Такую строку можно принять автоматически.

```text
reason_id=5, p_correct=0.48, decision=review
```

Такую строку лучше отдать на ручную проверку.

## Что важно понимать

`accept` не значит "модель гарантирует, что разметка правильная". Это значит:
на исторической проверке для этой причины был найден такой порог, при котором
авто-принятые строки целились в нужную точность, например `95%+`.

`review` тоже не значит "разметка неверная". Это значит, что валидатору не
хватило уверенности, и строку нужно оставить человеку.

## Как начать после копирования с GitHub

Этот раздел рассчитан на человека, который скачал проект с GitHub и запускает его на своем компьютере.

Важная техническая деталь: CLI запускается как Python-пакет `auto_classifier`, поэтому папка с кодом должна называться `auto_classifier`, а команды `python -m auto_classifier.cli ...` нужно выполнять из родительской папки.

### 1. Склонировать репозиторий

Рекомендуемый вариант:

```bash
mkdir AutoChecker-work
cd AutoChecker-work
git clone https://github.com/SadakovDmitry/AutoChecker.git auto_classifier
```

После этого структура должна быть такой:

```text
AutoChecker-work/
  auto_classifier/
    README.md
    USER_GUIDE.md
    cli.py
    data.py
    ...
```

Все команды ниже выполняются из папки `AutoChecker-work`, то есть из родительской папки для `auto_classifier`.

Если вы скачали ZIP-архив с GitHub, распакуйте его и переименуйте папку с кодом в `auto_classifier`, затем перейдите в родительскую папку:

```bash
mkdir AutoChecker-work
# распакуйте ZIP внутрь AutoChecker-work и переименуйте папку в auto_classifier
cd AutoChecker-work
```

### 2. Создать виртуальное окружение

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Если создание окружения было прервано через `Ctrl+C`, запустите `python3 -m venv .venv` еще раз. Иначе окружение может создаться без `pip` или без файла активации.

### 3. Установить зависимости

Быстрый вариант без embeddings:

```bash
pip install -r auto_classifier/requirements-minimal.txt
```

После этого обучайте модель с флагом `--no-embeddings`. Такой режим работает на `TF-IDF + LSA + retrieval + rules` и не требует тяжелых ML-зависимостей.

Полный вариант с sentence embeddings:

```bash
pip install -r auto_classifier/requirements.txt
```

Он ставит `sentence-transformers` и позволяет использовать embedding-модель `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Первая загрузка модели может занять время и потребует интернет.

Если `sentence-transformers` или embedding-модель недоступны, программа все равно работает в fallback-режиме без embeddings. Для явного запуска без embeddings используйте `--no-embeddings`.

### 4. Проверить, что CLI запускается

```bash
python -m auto_classifier.cli --help
```

Если команда показывает список подкоманд `prepare`, `train`, `verify`, `evaluate`, установка прошла нормально.

### 5. Запустить локальный интерфейс для аналитика

Самый простой сценарий теперь можно запускать без ручной сборки команд:

```bash
python -m auto_classifier.web_app
```

После запуска откройте в браузере:

```text
http://127.0.0.1:8787/
```

В интерфейсе нужно загрузить два файла:

- Excel с итерациями разметки: каждый лист - отдельная итерация, последний лист - новая неразмеченная итерация;
- файл с текстовками всех диалогов.

Дополнительно выберите тему/датасет, например `kasko_oformlenie`. Это нужно,
чтобы AutoChecker использовал стабильные `subreason_key`, если номера
подпричин менялись между итерациями.

На выходе интерфейс отдаст Excel-файл с листами:

| Лист | Что внутри |
| --- | --- |
| `marked_latest` | последняя итерация с колонками `авто_ответ`, `нужно_проверить_вручную`, `уверенность_p_correct` |
| `summary` | покрытие авторазметки по каждой подпричине |
| `risk_summary` | почему подпричина размечалась полностью или ушла в осторожный режим |
| `model_training` | качество и пороги обученных валидаторов |
| `metadata` | параметры запуска |

`авто_ответ` принимает значения:

| Значение | Что делать |
| --- | --- |
| `да` | можно принять автоматически |
| `нет` | можно принять автоматически, если подпричина прошла полный режим |
| `review` | нужно проверить руками |

В интерфейсе используется текущий лучший гибридный режим:

```text
threshold + max(history_latest, mean p_correct)
```

То есть для стабильных подпричин модель размечает `да/нет` через персональный
порог `p_correct`, а для рискованных подпричин ставит только уверенные `да` и
оставляет остальное в `review`.

### 6. Подготовить свои данные

Репозиторий не содержит реальные чаты, Excel-файлы и обученные модели. Нужно положить свои файлы локально, например:

```text
AutoChecker-work/
  my_data/
    labels.xlsx
    messages.xlsx
    new_classifier_results.xlsx
```

`my_data/` можно назвать как угодно. Главное - не коммитить туда реальные чаты, если проект хранится в GitHub.

### 7. Типовой рабочий сценарий

Собрать обучающий датасет из разметки и текстовок:

```bash
python -m auto_classifier.cli prepare \
  --labels "my_data/labels.xlsx" \
  --messages "my_data/messages.xlsx" \
  --output "my_data/prepared_training.csv"
```

Обучить модель без embeddings:

```bash
python -m auto_classifier.cli train \
  --data "my_data/prepared_training.csv" \
  --out "models/model_v1" \
  --target-precision 0.95 \
  --no-embeddings
```

Проверить новую выгрузку классификатора:

```bash
python -m auto_classifier.cli verify \
  --model "models/model_v1" \
  --input "my_data/new_classifier_results.xlsx" \
  --output "my_data/checked_results.xlsx"
```

Если ставили полный `requirements.txt` и хотите использовать embeddings, уберите `--no-embeddings` из команды `train`.

Важно: чтение `.xlsx` поддерживается даже без `openpyxl` через fallback, но для записи `.xlsx` нужен `openpyxl`. Если его нет, используйте `.csv` в `--output`.

## Формат данных

Для обучения, проверки и оценки нужен полный текст чата. Одних ссылок на чат и
комментариев недостаточно.

Минимальные колонки для обучения:

| Что нужно | Поддерживаемые названия |
| --- | --- |
| ID чата | `chat_id`, `comm_id`, `thread_id`, `id` |
| полный текст чата | `chat_text`, `text`, `communication`, `dialog`, `chat`, `текст`, `чат` |
| причина классификатора | `reason_id`, `reason_number`, `predicted_reason`, `reason`, `label`, `причина` |
| ручная проверка | `human_answer`, `да/нет`, `yes/no`, `answer`, `is_correct` |
| комментарий | `comment`, `комментарий`, `коментарий`, `комментарии` |

Для `verify` ручная проверка не нужна, потому что мы проверяем новые результаты
классификатора.

Если колонка с текстом называется иначе, передайте ее явно:

```bash
--text-column "full_chat"
```

## Как заполнять `да/нет`

Ручная оценка нормализуется так:

| Значение в таблице | Как используется |
| --- | --- |
| `да`, `yes`, `1`, `true` | правильная разметка, положительный пример |
| `нет`, `no`, `0`, `false` | ошибочная разметка, отрицательный пример |
| `да?` | считается как `да`, положительный пример |
| `нет?` | считается как `нет`, отрицательный пример |
| пусто | игнорируется в обучении и оценке |

Если классификатор вернул несколько причин, например `1,2`, строка будет
разбита на две проверки: отдельно для `1` и отдельно для `2`.

## Пример таблицы

CSV может выглядеть так:

```csv
comm_id,chat_text,reason_number,да/нет,комментарий
T1001,"client: хочу продлить полис
manager: уточните данные",2,да,"клиент сам спрашивает про автопродление"
T1002,"client: хочу оформить ОСАГО
bot: могу помочь с КАСКО",3,нет,"тема не про КАСКО"
```

Роли в тексте желательно оставлять в таком формате:

```text
client: текст клиента
manager: текст оператора
bot: текст бота
```

Поддерживаются роли:

- клиент: `client`, `клиент`, `customer`, `user`;
- оператор: `manager`, `operator`, `оператор`, `сотрудник`, `agent`, `support`;
- бот: `bot`, `бот`, `robot`.

Если роли есть, модель обучается в первую очередь на тексте клиента. Если
текста клиента нет, используется полный текст чата.

## Быстрый старт

### 0. Собрать обучающий файл из двух таблиц

Если ручная проверка лежит в одном файле, а тексты сообщений в другом, сначала
соберите единый датасет:

```bash
python3 -m auto_classifier.cli prepare \
  --labels "labels.xlsx" \
  --messages "messages.xlsx" \
  --output "prepared_training.csv"
```

Файл `--labels` должен содержать:

- `comm_id` / `chat_id` - id диалога;
- `reason_numb` / `reason_id` / `reason_number` - причина классификатора;
- `да/нет` - ручная проверка;
- `комментарий` - необязательно.

Файл `--messages` может быть в формате "одно сообщение = одна строка":

| Колонка | Смысл |
| --- | --- |
| `ID_diologa` | id диалога |
| `Vremya` | время сообщения |
| `Kto` | отправитель: `client`, `manager`, `bot` |
| `Soobschenie` | текст сообщения |

Также поддерживается новый формат "один диалог = одна строка":

| Колонка | Смысл |
| --- | --- |
| `ID_diologa` | id диалога |
| `Kolichestvo_soobscheniy` | количество сообщений |
| `Pervoe_soobschenie` | время первого сообщения |
| `Poslednee_soobschenie` | время последнего сообщения |
| `Dialog_polnostyu` | полный диалог в формате `время | CLIENT/BOT/MANAGER | текст` |

Команда сгруппирует сообщения или разберет `Dialog_polnostyu` и сделает
единый `chat_text`:

```text
client: текст клиента
bot: текст бота
manager: текст оператора
```

Именно такой формат лучше всего подходит валидатору, потому что он умеет
отделять слова клиента от слов оператора и бота.

Если нужно подготовить файл для `verify`, где еще нет ручного `да/нет`, добавьте
флаг:

```bash
python3 -m auto_classifier.cli prepare \
  --labels "new_classifier_results.xlsx" \
  --messages "messages.xlsx" \
  --output "prepared_verify.csv" \
  --allow-unlabeled
```

### 1. Обучить валидатор

Безопасный режим по умолчанию: модель автоматически ставит только `да`, а все
остальное отправляет в `review`.

```bash
python3 -m auto_classifier.cli train \
  --data "prepared_training.csv" \
  --text-column chat_text \
  --out auto_classifier/models/kasko_v1 \
  --target-precision 0.95 \
  --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Экспериментальный режим с автоматическим `нет` включается отдельным флагом:

```bash
python3 -m auto_classifier.cli train \
  --data "prepared_training.csv" \
  --out auto_classifier/models/kasko_v1_yesno \
  --target-precision 0.95 \
  --target-no-precision 0.97 \
  --max-auto-no-p-correct 0.04 \
  --enable-auto-no
```

`--target-precision` управляет автоматическим `да`.
`--target-no-precision` управляет автоматическим `нет`; обычно его стоит делать строже, например `0.97`, чтобы модель ставила `нет` только в почти очевидных случаях.
`--max-auto-no-p-correct` - дополнительный предохранитель: даже если обучающий подбор разрешил более широкий порог, `нет` будет ставиться только при `p_correct` не выше этого значения.

После этого в `verify/evaluate` появятся решения:

```text
auto_yes -> auto_answer = да
auto_no  -> auto_answer = нет
review   -> auto_answer = review
```

Если модель обучена без `--enable-auto-no`, решений `auto_no` не будет.

Если нужно быстро проверить пайплайн без embeddings:

```bash
python3 -m auto_classifier.cli train \
  --data "prepared_training.csv" \
  --text-column chat_text \
  --out auto_classifier/models/kasko_v1 \
  --target-precision 0.95 \
  --no-embeddings
```

После обучения появится папка модели:

```text
auto_classifier/models/kasko_v1/
  model.joblib
  training_summary.csv
```

`training_summary.csv` показывает, что произошло по каждой причине:

- `reason_id` - номер причины;
- `threshold` - выбранный порог;
- `threshold_precision` - точность авто-приемки на внутренней проверке;
- `threshold_coverage` - доля строк, которую можно было принять;
- `n_samples`, `n_positive`, `n_negative` - размер обучения;
- `warnings` - проблемы вроде `low_data`.

### 2. Проверить новые результаты классификатора

```bash
python3 -m auto_classifier.cli verify \
  --model auto_classifier/models/kasko_v1 \
  --input new_classifier_results.xlsx \
  --output checked_results.xlsx
```

Если нет `openpyxl`, пишите в CSV:

```bash
python3 -m auto_classifier.cli verify \
  --model auto_classifier/models/kasko_v1 \
  --input new_classifier_results.xlsx \
  --output checked_results.csv
```

### 3. Оценить качество на размеченной выборке

```bash
python3 -m auto_classifier.cli evaluate \
  --model auto_classifier/models/kasko_v1 \
  --data validation.xlsx \
  --output report.xlsx
```

Для `.xlsx` отчета создаются листы:

- `summary` - качество по причинам;
- `predictions` - все строки с предсказаниями валидатора;
- `accepted_errors` - ошибки среди автоматически принятых строк.

Если output в `.csv`, программа создаст три файла:

```text
report_summary.csv
report_predictions.csv
report_accepted_errors.csv
```

## Как читать результат `verify`

Главные колонки:

| Колонка | Что значит |
| --- | --- |
| `p_correct` | итоговая вероятность корректности разметки |
| `threshold` | порог для авто-приемки этой причины |
| `decision` | `accept`, если `p_correct >= threshold`, иначе `review` |
| `p_lsa` | вероятность baseline-модели `TF-IDF + LSA` |
| `p_embedding` | вероятность embedding-модели, если она доступна |
| `nearest_positive_chat_id` | самый похожий исторический правильный пример |
| `nearest_positive_score` | похожесть на правильный пример |
| `nearest_negative_chat_id` | самый похожий исторический ошибочный пример |
| `nearest_negative_score` | похожесть на ошибочный пример |
| `rule_flags` | заметные rule-признаки |
| `validator_warnings` | предупреждения по причине |

Практическое чтение:

- `decision=accept` - можно не проверять руками, если вас устраивает целевая
  точность модели на `evaluate`;
- `decision=review` - оставить на ручную проверку;
- высокий `nearest_negative_score` - чат похож на прошлую ошибку, стоит
  посмотреть внимательно;
- `operator_only_keywords` - ключевые слова нашлись у оператора/бота, но не у
  клиента;
- `no_client_text` - программа не нашла реплики клиента по ролям.

## Как работает внутри

### 1. Нормализация таблиц

Программа читает `.csv`, `.xlsx`, `.json`, `.jsonl`, приводит названия колонок к
единым именам и разворачивает multi-label ответы вроде `1,2`.

Для обучения оставляет только строки, где ручная оценка однозначная:

```text
да  -> 1
нет -> 0
```

Для текущего сравнения `да?` и `нет?` участвуют в обучении как обычные `да` и `нет`.

### 2. Разделение ролей

Если в `chat_text` есть роли, текст делится на:

- `client_text`;
- `operator_text`;
- `bot_text`;
- `full_text`.

Основной текст для модели:

```text
model_text = client_text, если он не пустой
model_text = full_text, если клиентскую часть выделить не удалось
```

Это важно, потому что классификатор часто ошибается на ответах оператора:
оператор пишет "можно оформить страховку багажа", а клиент сам это не просил.

### 3. Отдельный валидатор на каждую причину

Для каждой причины обучается отдельная бинарная модель:

```text
для reason_id=5:
  1 = классификатор правильно поставил причину 5
  0 = классификатор ошибочно поставил причину 5
```

То есть модель не выбирает новую причину. Она отвечает на вопрос:

```text
"Похоже ли, что эта конкретная причина действительно подходит этому чату?"
```

### 4. TF-IDF + LSA baseline

Сначала строится прозрачный текстовый baseline:

```text
client_text -> TF-IDF -> LSA -> LogisticRegression -> p_lsa
```

Он хорошо ловит устойчивые формулировки и ключевые конструкции:

- "полис продлится сам";
- "хочу рассчитать КАСКО";
- "у другой страховой дешевле";
- "нужна страховка для автокредита".

### 5. Sentence embeddings

Если доступен `sentence-transformers`, дополнительно строится смысловой сигнал:

```text
client_text -> sentence embeddings -> LogisticRegression -> p_embedding
```

Embeddings нужны, чтобы находить похожие по смыслу формулировки, даже если
слова другие.

Если библиотека или модель недоступны, программа пишет предупреждение
`embedding_unavailable` и продолжает работу на TF-IDF/LSA.

### 6. Retrieval-сигналы

Для нового чата считается похожесть на исторические примеры этой же причины:

```text
sim_pos = похожесть на ближайший правильный пример
sim_neg = похожесть на ближайший ошибочный пример
sim_margin = sim_pos - sim_neg
```

Если чат очень похож на прошлую ошибку, это сильный сигнал отправить его в
`review`.

### 7. Rule features

YAML-правила не являются жесткими фильтрами. Они превращаются в числовые
признаки:

- сколько ключевых слов найдено в клиентской части;
- сколько найдено в полном тексте;
- сколько найдено у оператора;
- есть ли ключевые слова только у оператора/бота;
- есть ли обязательные слова именно у клиента.

Это помогает ловить кейсы из твоих комментариев: "сработало только на ответ
оператора", "клиент сам не писал о проблеме", "тема вообще про другой продукт".

### 8. Финальная модель

Все сигналы собираются в финальный слой:

```text
p_lsa
p_embedding
sim_pos
sim_neg
sim_margin
client_keyword_hits
operator_only_hits
client_text_share
...
  -> LogisticRegression(class_weight="balanced")
  -> p_correct
```

Если данных достаточно, вероятность калибруется через
`CalibratedClassifierCV`. Если данных мало, используется обычная вероятность и
в отчет пишется warning.

### 9. Подбор порогов

Порог выбирается отдельно для каждой причины.

Алгоритм:

1. На обучающей выборке строятся out-of-fold предсказания.
2. Для разных порогов считается precision среди строк, которые были бы
   автоматически приняты.
3. Выбирается порог с максимальным coverage при условии:

```text
accepted_precision >= --target-precision
```

По умолчанию:

```text
--target-precision 0.95
```

Если по причине нельзя достичь нужной точности, порог ставится выше `1.0`.
Тогда все новые строки этой причины получают `review`. Это нормальное
поведение: лучше не принимать автоматически причину, где валидатор не научился
быть достаточно точным.

## Optional rules YAML

Файл правил можно передать так:

```bash
python3 -m auto_classifier.cli train \
  --data data.csv \
  --out auto_classifier/models/demo \
  --rules auto_classifier/configs/default_rules.yaml
```

Пример:

```yaml
reasons:
  "5":
    keywords:
      - страховка багажа
      - подписка pro
      - багаж
    client_required_keywords:
      - багаж
    operator_only_keywords:
      - оформить страховку багажа
      - можно подключить страховку багажа
```

Как это читать:

- `keywords` - общие подсказки по теме;
- `client_required_keywords` - слова, которые особенно полезно видеть именно у
  клиента;
- `operator_only_keywords` - фразы, которые подозрительны, если они есть только
  у оператора/бота.

Важно: эти правила не говорят модели "обязательно принять" или "обязательно
отклонить". Они дают дополнительные признаки финальной модели.

## Как дообучать

Рекомендуемый цикл:

1. Обучить модель на исторических `да/нет`.
2. Прогнать `verify` на новых результатах классификатора.
3. Автоматически принять только `decision=accept`.
4. Руками проверить `decision=review`.
5. Добавить ручную проверку в обучающую таблицу.
6. Переобучить модель в новую папку, например:

```text
auto_classifier/models/kasko_v2
auto_classifier/models/kasko_v3
```

Старые модели лучше не перезаписывать, чтобы можно было сравнивать качество
итераций.

## Почему модель может принять мало строк

Это ожидаемо в нескольких случаях:

- мало размеченных примеров по причине;
- мало ошибок или мало правильных примеров, поэтому нечему учиться;
- причина слишком широкая и внутри нее разные сценарии;
- историческая разметка шумная;
- `--target-precision` слишком высокий;
- embeddings недоступны и остался только fallback.

Для первой версии это не проблема. Цель - начать с надежной авто-приемки части
строк, а не сразу заменить ручную проверку полностью.

## Что делать, если ошибка "нужен полный текст чата"

Ошибка выглядит так:

```text
Для валидатора нужен полный текст чата: добавьте колонку chat_text или передайте --text-column
```

Это значит, что в таблице нет текста самого диалога. Ссылки на чат, `comm_id`,
`reason_number`, `да/нет` и комментарии аналитика недостаточны: модель должна
видеть, что писал клиент, оператор и бот.

Нужно выгрузить таблицу с колонкой, где лежит полный текст коммуникации, и
назвать ее `chat_text` или передать название через `--text-column`.

## Что смотреть после обучения

Откройте:

```text
auto_classifier/models/<model_name>/training_summary.csv
```

Хороший признак:

- у причины есть и `n_positive`, и `n_negative`;
- `warnings` пустой;
- `threshold_coverage` больше `0`;
- `threshold_precision` не ниже целевого значения.

Плохой или осторожный признак:

- `low_data` - мало данных;
- `no_threshold_for_target_precision` - не удалось подобрать надежный порог;
- `threshold = 1.000001` - причина не будет приниматься автоматически.

## Что смотреть после evaluate

В `summary` важны:

- `coverage` - какую долю строк модель автоматически принимает;
- `accepted_precision` - точность среди принятых;
- `overall_positive_rate` - исходная доля правильных ответов классификатора.

Главный критерий:

```text
accepted_precision >= target_precision
```

Если `coverage` маленький, но `accepted_precision` высокий, валидатор работает
консервативно. Это нормально для первой продовой версии.

Лист `accepted_errors` особенно важен: это ошибки, которые валидатор все-таки
пропустил в `accept`. По ним стоит добавить YAML-признаки или больше обучающих
примеров.

## Ограничения

- Валидатор проверяет разметку, но не исправляет ее на другую причину.
- Для каждой причины нужны и правильные, и неправильные примеры.
- Очень маленькие причины будут уходить в `review`.
- Качество зависит от того, насколько стабильно размечены исторические данные.
- Если номера причин менялись между итерациями, используйте `subreason_key`,
  иначе модель может смешать разные смыслы под одним `reason_id`.
- Если роли в чате не размечены, модель хуже отделяет слова клиента от слов
  оператора.
- Если в таблице нет `chat_text`, обучение и проверка невозможны.

## Версии Подпричин

`reason_id` - это номер причины внутри конкретного промпта. Если в новой
итерации причины переименовали, объединили или поменяли порядок, один и тот же
номер может означать разные вещи. Для этого есть YAML-словарь версий
подпричин.

Пример:

```yaml
datasets:
  kasko_uregulirovanie:
    files:
      - "КАСКО - Урегулирование убытков.xlsx"
    iterations:
      "итерация 1":
        reasons:
          "1": unclear_status_and_communication
          "4": unclear_status_and_communication
          "7": unclear_status_and_communication
      "итерация 2":
        reasons:
          "1": unclear_status_and_communication
```

Так старые причины `1`, `4`, `7` обучаются как одна стабильная подпричина
`unclear_status_and_communication`.

Подготовить файл с добавленным `subreason_key`:

```bash
python3 -m auto_classifier.cli prepare \
  --labels "local_data/labels_normalized/КАСКО - Урегулирование убытков.xlsx" \
  --messages "local_data/messages/текстовки-КАСКО-Урегулирования.xlsx" \
  --subreason-map "auto_classifier/configs/subreason_versions.example.yaml" \
  --output "local_data/prepared/kasko_uregulirovanie_mapped.csv"
```

Обучаться по стабильным ключам:

```bash
python3 -m auto_classifier.cli train \
  --data "local_data/prepared/kasko_uregulirovanie_mapped.csv" \
  --subreason-map "auto_classifier/configs/subreason_versions.example.yaml" \
  --group-by-subreason-key \
  --out "models/kasko_uregulirovanie_stable"
```

Если для строки нет явного маппинга, ей ставится безопасный ключ вида
`unmapped::<dataset>::<iteration>::<reason_id>`. Такая строка не смешается с
другой итерацией случайно.

## Короткая памятка команд

Обучить:

```bash
python3 -m auto_classifier.cli train \
  --data "data/*.xlsx" \
  --out auto_classifier/models/model_v1 \
  --target-precision 0.95
```

Обучить без embeddings:

```bash
python3 -m auto_classifier.cli train \
  --data "data/*.xlsx" \
  --out auto_classifier/models/model_v1 \
  --target-precision 0.95 \
  --no-embeddings
```

Проверить новые результаты:

```bash
python3 -m auto_classifier.cli verify \
  --model auto_classifier/models/model_v1 \
  --input new_results.xlsx \
  --output checked_results.xlsx
```

Оценить качество:

```bash
python3 -m auto_classifier.cli evaluate \
  --model auto_classifier/models/model_v1 \
  --data validation.xlsx \
  --output report.xlsx
```
# AutoChecker
