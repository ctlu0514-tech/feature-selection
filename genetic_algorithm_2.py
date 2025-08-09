import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification

class GAFeatureSelector:
    def __init__(self, X, y, population_size=20, generations=10,
                 crossover_prob=0.7, mutation_prob=0.02, beta=0.01, k=6,
                 test_size=0.3, random_state=None):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.beta = beta  # penalty weight
        self.k = k  # k for kNN
        self.test_size = test_size
        self.random_state = random_state
        self.n_features = X.shape[1]
        # split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

    def _initialize_population(self):
        # binary matrix: population_size x n_features
        return np.random.randint(0, 2, size=(self.population_size, self.n_features))

    def _fitness(self, chromosome):
        # 如果没有选中特征，返回较差适应度
        if np.sum(chromosome) == 0:
            return 1.0 + self.beta * 0
        # 根据染色体选特征
        idx = np.where(chromosome == 1)[0]
        X_train_sel = self.X_train[:, idx]
        X_test_sel = self.X_test[:, idx]
        # 训练 kNN
        clf = KNeighborsClassifier(n_neighbors=self.k)
        clf.fit(X_train_sel, self.y_train)
        acc = clf.score(X_test_sel, self.y_test)
        mcr = 1 - acc  # misclassification rate
        # 目标函数 mcr * (1 + beta * n_selected)
        return mcr * (1 + self.beta * len(idx))

    def _select(self, population, fitnesses):
        # 轮盘赌选择，fitness lower is better, so invert
        inv_fitness = 1 / (fitnesses + 1e-6)
        probs = inv_fitness / np.sum(inv_fitness)
        indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=probs)
        return population[indices]

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, self.n_features - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, chromosome):
        for i in range(self.n_features):
            if np.random.rand() < self.mutation_prob:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def run(self):
        # 初始化种群
        population = self._initialize_population()
        best_chromosome = None
        best_fitness = np.inf

        for gen in range(self.generations):
            # 计算适应度
            fitnesses = np.array([self._fitness(ch) for ch in population])
            # 记录最优
            idx_best = np.argmin(fitnesses)
            if fitnesses[idx_best] < best_fitness:
                best_fitness = fitnesses[idx_best]
                best_chromosome = population[idx_best].copy()
            print(f"Generation {gen+1}: Best fitness = {best_fitness:.4f}, Features selected = {np.sum(best_chromosome)}")
            # 选择
            selected = self._select(population, fitnesses)
            # 生成新一代
            next_pop = []
            for i in range(0, self.population_size, 2):
                p1, p2 = selected[i], selected[i+1]
                c1, c2 = self._crossover(p1, p2)
                next_pop.append(self._mutate(c1))
                next_pop.append(self._mutate(c2))
            population = np.array(next_pop)

        # 输出最优结果
        selected_features = np.where(best_chromosome == 1)[0]
        print("Best chromosome:", best_chromosome)
        print("Selected feature indices:", selected_features)
        print("Test accuracy with selected features:", 1 - best_fitness / (1 + self.beta * len(selected_features)))
        return best_chromosome, selected_features

if __name__ == '__main__':
    # # 示例：使用鸢尾花数据集
    # data = load_iris()
    # X, y = data.data, data.target
    # ga_fs = GAFeatureSelector(X, y, population_size=20, generations=10,
    #                           crossover_prob=0.7, mutation_prob=0.02,
    #                           beta=0.01, k=6, test_size=0.3,
    #                           random_state=42)
    # best_chrom, features = ga_fs.run()

# 高维小样本示例：合成基因表达数据
    X, y = make_classification(n_samples=150, n_features=2000,
                               n_informative=50, n_redundant=450,
                               n_classes=3, random_state=42)
    ga_fs = GAFeatureSelector(X, y, random_state=42)
    best_chrom, features = ga_fs.run()
