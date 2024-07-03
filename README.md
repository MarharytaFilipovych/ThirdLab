# Third Lab
## Які основні етапи включає SVD розклад і як цей метод можна застосувати до вирішення задачі підбору рекомендацій для певного користувача?

Сингулярний розклад матриці А розміру m*n, яка складена з дійсних або комплексних чисел, буде розкладанням на множники у вигляді 𝑈𝛴𝑉, де 𝑈 — матриця розміру m*m, 𝛴 буде m*n діагональною матрицею з не від'ємними дійсними числами на діагоналі, і 
𝑉 буде матрицею розміру n*n. Діагональні елементи 𝜎 матриці 𝛴 відомі як сингулярні значення. Стовпчики 𝑈 та 𝑉 називаються ліво-сингулярними векторами та право-сингулярними векторами матриці 𝑀, відповідно.
__Сингулярний розклад матриці можна обчислити за допомогою наступних спостережень:__
* Ліво-сингулярні вектори А є множиною ортонормованих головних векторів ААТ.
* Право-сингулярні вектори А є множиною ортонормованих головних векторів АТА.
* Не нульові сингулярні значення M (знаходяться на діагоналі Σ) є квадратними коренями не нульових власних значень як ААТ, так і ААТ.

SVD є потужною технікою, що використовується в системах рекомендацій для створення персоналізованих рекомендацій для користувачів.
Розкладаючи матрицю user-item а компоненти, SVD дозволяє знаходити приховані представлення користувачів і товарів,
які можна використовувати для різних способів надання рекомендацій, таких як знаходження схожих користувачів або товарів, а також прогнозування того,
як користувач оцінить незнайомий йому товар. SVD також корисний у роботі з відсутніми даними, що є поширеною проблемою в системах рекомендацій.

Ми можемо виконати стиснення даних, витягуючи важливу інформацію з наявних даних. Цей процес також відомий як зменшення розмірності, і це одне з найпоширеніших застосувань сингулярного розкладу.
Ключем до зменшення розмірності є те, що перші кілька стовпців 𝑈, його відповідні власні значення у Σ та перші кілька рядків 𝑉𝑇 містять найбільшу кількість інформації про матрицю 𝐴.

## В яких сферах застосовується SVD?

* **зменшення розмірності**: чим більше даних, тим важче їх обробляти та візуалізувати. Використовуючи SVD, ми можемо зменшити розмірність наших даних, зберігаючи при цьому більшість їх варіацій. Це принцип, що лежить в основі технік, таких як аналіз головних компонент (PCA).
* **зменшення шуму**: у багатьох випадках дані містять шум. SVD допомагає розкладати матрицю таким чином, щоб відокремити сигнал від шуму. 
* **системи рекомендацій**: алгоритм розкладає матрицю взаємодій користувач-товар для прогнозування відсутніх значень, що представляють оцінки користувачів.
* **виділення важливих ознак**

## Як вибір параметра k у SVD розкладі впливає на результат?

k = number of singular values and singular vectors to compute.
Чим більше значення k (ближче до початкової кількості рядків), тим точніше відновлена матриця буде відповідати оригінальній. При занадто малому значенні k частина даних буде втрачена, що може призвести до погіршення точності відновлення цих даних
При малих значеннях k можуть з'явитися значні похибки, а при дуже великих - система може втратити здатність до узагальнення.
Оптимальне значення k зазвичай визначається експериментально.

## Які основні переваги та недоліки має SVD?

**Переваги**: Спрощує дані, видаляє шум, може покращити результати алгоритму.
**Недоліки**: Трансформовані дані можуть бути важкими для розуміння або неточними.

Джерела:
https://uk.wikipedia.org/wiki/%D0%A1%D0%B8%D0%BD%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%BD%D0%B8%D0%B9_%D1%80%D0%BE%D0%B7%D0%BA%D0%BB%D0%B0%D0%B4_%D0%BC%D0%B0%D1%82%D1%80%D0%B8%D1%86%D1%96
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html
https://www.linkedin.com/advice/3/how-can-you-use-singular-value-decomposition-machine#:~:text=Singular%20value%20decomposition%20(SVD)%20is,extraction%2C%20and%20latent%20factor%20analysis
https://medium.com/@ritik_gupta/how-singular-value-decomposition-svd-is-used-in-recommendation-systems-clearly-explained-201b24e175db
https://jaketae.github.io/study/svd/