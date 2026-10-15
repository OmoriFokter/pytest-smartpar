# pytest-smartpar

Плагін pytest, що реалізує метод паралельного виконання модульних тестів у CI
з урахуванням спільних файлів, спільного стану в процесі та тривалості тестів.

> **Статус:** у розробці (фаза 1 – каркас). Режими `--smartpar-*` ще не реалізовані
> і при використанні завершуються з повідомленням про помилку. Повна документація
> та інструкція відтворення експерименту з'являться у фазі 9 (див. `SPEC.md`).

**English summary:** pytest plugin implementing a method for parallel execution of
unit tests in CI (file-conflict detection, order-dependency detection, LPT scheduling).
Work in progress.
