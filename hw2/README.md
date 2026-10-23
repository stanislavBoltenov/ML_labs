# ДЗ 2. kNN и другие классификаторы

Результаты на тесте:

| Модель | Ошибка | Balanced accuracy | macro-F1 |
|---|---|---|---|
| Всегда excellent | 0.505 | 0.167 | 0.110 |
| kNN, k = 51 | 0.334 | 0.251 | 0.235 |
| Logistic Regression | 0.340 | 0.309 | 0.321 |
| SVC (rbf) | 0.315 | 0.265 | 0.253 |
| Random Forest | 0.270 | 0.343 | 0.377 |
| Gradient Boosting | 0.290 | 0.417 | 0.444 |
| Random Forest, веса классов | 0.315 | 0.476 | 0.435 |
| Gradient Boosting, веса классов | 0.397 | 0.546 | 0.410 |

Данные берутся из `../craigslist_dataset/` (см. README в корне).
