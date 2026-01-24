# seismocorr/core/traditional_inversion/adaptive_ga.py
'''
自适应遗传算法实现（反演层厚度、横波速度）
适配：分层边界列表长度定义层数，最后1层强制为半空间（thickness=0）
'''
from typing import Optional, List, Callable, Dict, Union, Tuple

import numpy as np
from scipy.stats import uniform

from .types import GAConfig, InversionParams, DispersionCurve
from .utils import (
    validate_inversion_params,
    calculate_residual,
    calculate_fitness,
    denormalize_params
)

# GA默认配置（适配分层边界格式，默认9层，与你的需求一致）
DEFAULT_GA_CONFIG = {
    "pop_size": 60,          # 种群大小
    "max_generations": 80,  # 最大迭代代数
    "crossover_rate": 0.8,   # 初始交叉率
    "mutation_rate": 0.1,    # 初始变异率
    "adapt_strategy": "exponential",  # 自适应策略: linear/exponential
    "elitism_ratio": 0.1,    # 精英保留比例
    "param_bounds": {        # 分层边界格式：列表长度=层数，最后1层thickness强制0
        "thickness": [
            (0.0, 5.0), (0.0, 5.0), (0.0, 6.0), (0.0, 6.0),
            (0.0, 8.0), (0.0, 8.0), (0.0, 10.0), (0.0, 10.0),
            (0.0, 0.0)
        ],
        "vs": [
            (2.0, 3.0), (2.0, 3.0), (2.5, 3.5), (2.5, 3.5),
            (3.0, 4.0), (3.0, 4.0), (3.5, 4.5), (3.5, 4.5),
            (3.0, 4.0)
        ]
    }
}


class AdaptiveGA:
    """
    自适应遗传算法类，用于层厚度、横波速度等反演参数的全局搜索
    层数规则：由param_bounds["thickness"]的列表长度自动决定，最后1层强制为半空间
    自适应策略：根据种群适应度分布动态调整交叉率和变异率
    适应度高的个体使用较小的交叉/变异率，保持优秀基因；
    适应度低的个体使用较大的交叉/变异率，促进全局探索。
    """

    def __init__(self, config: Optional[GAConfig] = None):
        """
        初始化自适应遗传算法
        Args:
            config: GA配置字典，未指定则使用默认配置
        """
        # ========== 修正点1：深合并配置，避免param_bounds被完全覆盖 ==========
        self.config = DEFAULT_GA_CONFIG.copy()
        if config is not None:
            # 浅合并非param_bounds的配置项
            for k, v in config.items():
                if k != "param_bounds":
                    self.config[k] = v
                # 深合并param_bounds（仅覆盖用户指定的key，内层列表保留默认）
                else:
                    if "param_bounds" not in self.config:
                        self.config["param_bounds"] = {}
                    for sub_k, sub_v in v.items():
                        self.config["param_bounds"][sub_k] = sub_v
        
        self.pop_size = self.config["pop_size"]
        self.max_generations = self.config["max_generations"]
        self.crossover_rate = self.config["crossover_rate"]
        self.mutation_rate = self.config["mutation_rate"]
        self.adapt_strategy = self.config["adapt_strategy"]
        self.elitism_size = int(self.pop_size * self.config["elitism_ratio"])
        self.param_bounds = self.config["param_bounds"]

        # 核心：从thickness边界列表长度获取层数，强制校验vs长度匹配
        self._validate_bounds()
        self.n_layers = len(self.param_bounds["thickness"])  # 层数由列表长度定

        # 正演函数（外部注入，适配thickness参数）
        self.forward_model: Optional[Callable[[InversionParams], DispersionCurve]] = None
        # 种群和适应度记录（新增：缓存当前代适应度，避免重复计算）
        self.population: List[InversionParams] = []
        self.fitness_history: List[float] = []
        self.current_fitness: List[float] = []  # 缓存当前代所有个体的适应度

    def _validate_bounds(self) -> None:
        """
        校验分层边界的合法性（核心：保证层数匹配、格式正确）
        1. thickness和vs必须是列表，且长度一致（层数相同）
        2. 列表中每个元素必须是2元素的二元组（min, max）
        3. 层数至少为2（1层实际地层 + 1层半空间，避免无意义反演）
        """
        # 检查是否包含必要键
        for key in ["thickness", "vs"]:
            if key not in self.param_bounds:
                raise KeyError(f"param_bounds必须包含{key}键，且值为二元组列表")
            # 检查是否为列表
            if not isinstance(self.param_bounds[key], list):
                raise TypeError(f"{key}必须是二元组列表（如[(0,5), (0,5)]），当前为{type(self.param_bounds[key])}")
            # 检查每个元素是否为2元素二元组
            for idx, bound in enumerate(self.param_bounds[key]):
                if not isinstance(bound, (tuple, list)) or len(bound) != 2:
                    raise ValueError(f"{key}第{idx+1}个元素必须是2元素二元组（如(0.0,5.0)），当前为{bound}")
            # 检查数值是否合法（min<=max）
            for idx, (lower, upper) in enumerate(self.param_bounds[key]):
                if lower > upper:
                    raise ValueError(f"{key}第{idx+1}个元素的下界({lower})不能大于上界({upper})")

        # 强制检查thickness和vs的列表长度一致（层数匹配）
        thk_len = len(self.param_bounds["thickness"])
        vs_len = len(self.param_bounds["vs"])
        if thk_len != vs_len:
            raise ValueError(f"thickness列表长度({thk_len})与vs列表长度({vs_len})不一致，层数必须匹配")

        # 层数至少为2
        if thk_len < 2:
            raise ValueError(f"层数必须≥2（1层实际地层+1层半空间），当前为{thk_len}层")

    # ========== 新增：校验观测曲线格式 ==========
    def _validate_observed_curve(self, observed_curve: DispersionCurve) -> None:
        """校验观测频散曲线的合法性"""
        if not isinstance(observed_curve, (tuple, list)) or len(observed_curve) != 2:
            raise ValueError("观测曲线必须是(频率数组, 相速度数组)的二元组")
        freq, vel = np.asarray(observed_curve[0]), np.asarray(observed_curve[1])
        if freq.ndim != 1 or vel.ndim != 1:
            raise ValueError("观测曲线的频率/速度必须是一维数组")
        if len(freq) != len(vel):
            raise ValueError("观测曲线的频率和速度数组长度不一致")
        if np.all(np.isnan(vel)) or np.all(np.isinf(vel)):
            raise ValueError("观测曲线无有效速度值（全为nan/inf）")

    def set_forward_model(self, forward_func: Callable[[InversionParams], DispersionCurve]) -> None:
        """
        设置频散曲线正演函数（注入外部成熟的正演库，需支持thickness参数）
        Args:
            forward_func: 正演函数，输入反演参数（thickness/vs），输出频散曲线 (频率, 相速度)
        """
        self.forward_model = forward_func

    def _init_population(self) -> None:
        """
        初始化种群（归一化区间[0,1]）
        核心：按分层边界列表长度生成对应层数，强制thickness最后1层为0（半空间）
        """
        self.population = []
        param_keys = ["thickness", "vs"]  # 固定反演参数

        for _ in range(self.pop_size):
            individual = {}
            for key in param_keys:
                # 初始化对应层数的归一化参数数组
                ind_param = np.zeros(self.n_layers, dtype=np.float64)
                # 遍历每层边界，逐个解包生成随机归一化值
                for i in range(self.n_layers):
                    lower, upper = self.param_bounds[key][i]
                    # 归一化区间[0,1]随机生成
                    ind_param[i] = uniform.rvs(loc=0, scale=1)
                # 强制：thickness最后1层设为0（归一化值），实现半空间，无视配置
                if key == "thickness":
                    ind_param[-1] = 0.0
                individual[key] = ind_param
            self.population.append(individual)

    # ========== 修正点2：复用缓存的适应度，避免重复计算 ==========
    def _compute_population_fitness(self) -> None:
        """计算并缓存当前种群的所有个体适应度（仅计算一次/代）"""
        self.current_fitness = [self._calculate_individual_fitness(ind) for ind in self.population]

    def _adjust_params(self, generation: int) -> Tuple[float, float]:
        """
        自适应调整交叉率和变异率
        Args:
            generation: 当前迭代代数
        Returns:
            cross_rate: 调整后的交叉率
            mut_rate: 调整后的变异率
        """
        # 复用缓存的适应度（避免重复计算）
        if not self.current_fitness:
            self._compute_population_fitness()
        avg_fitness = np.mean(self.current_fitness)
        max_fitness = np.max(self.current_fitness)

        # 根据自适应策略调整
        if self.adapt_strategy == "exponential":
            # 指数衰减：迭代后期逐步降低探索性
            decay = np.exp(-generation / self.max_generations)
            cross_rate = self.crossover_rate * decay
            mut_rate = self.mutation_rate * decay
        else:  # linear
            # 线性衰减
            decay = 1 - (generation / self.max_generations)
            cross_rate = self.crossover_rate * decay
            mut_rate = self.mutation_rate * decay

        # 适应度差异调整：适应度越高，交叉/变异率越低
        if max_fitness - avg_fitness > 1e-6:  # 避免除零
            cross_rate = cross_rate * (avg_fitness / max_fitness)
            mut_rate = mut_rate * (avg_fitness / max_fitness)

        # 保证率值在合理范围
        cross_rate = np.clip(cross_rate, 0.5, 0.9)
        mut_rate = np.clip(mut_rate, 0.01, 0.2)
        return cross_rate, mut_rate

    def _calculate_individual_fitness(self, individual: InversionParams) -> float:
        """
        计算单个个体的适应度（基于频散曲线残差）
        Args:
            individual: 归一化的个体参数（thickness/vs）
        Returns:
            fitness: 适应度值（残差越小，适应度越高）
        """
        if self.forward_model is None:
            raise RuntimeError("请先通过set_forward_model设置正演函数")

        # 反归一化：将[0,1]的归一化值还原为物理区间值
        denorm_ind = denormalize_params(individual, self.param_bounds)
        # ========== 修正点3：校验反归一化后的参数合法性 ==========
        validate_inversion_params(denorm_ind)
        # 正演生成合成频散曲线
        synthetic_curve = self.forward_model(denorm_ind)
        # 计算残差（合成曲线与观测曲线的误差）
        _, residual_norm = calculate_residual(self.observed_curve, synthetic_curve)
        # 计算适应度（残差越小，适应度越高）
        fitness = calculate_fitness(residual_norm)
        return fitness

    def _selection(self) -> List[InversionParams]:
        """
        选择操作：轮盘赌选择 + 精英保留
        彻底修复：对最终传入p参数的非精英子概率列表单独归一化，解决精度偏差问题
        Returns:
            selected_pop: 选择后的种群
        """
        # 复用缓存的适应度（避免重复计算）
        if not self.current_fitness:
            self._compute_population_fitness()
        fitness_list = self.current_fitness
        
        # 精英保留：保留适应度最高的前elitism_size个个体
        elite_indices = np.argsort(fitness_list)[-self.elitism_size:]
        elite = [self.population[i] for i in elite_indices]
    
        # 轮盘赌选择：基础概率计算（无需提前归一化，后续针对子列表处理）
        total_fitness = sum(fitness_list)
        if total_fitness < 1e-12:  # 所有适应度接近0时，等概率选择
            prob_list = np.ones(self.pop_size) / self.pop_size
        else:
            prob_list = np.array([f / total_fitness for f in fitness_list])  # 转numpy数组，精度更稳定
    
        # 提取非精英个体的索引和对应的子概率列表（报错的核心关联对象）
        non_elite_indices = [i for i in range(self.pop_size) if i not in elite_indices]
        non_elite_prob = prob_list[non_elite_indices]  # numpy切片，比纯列表更稳定
    
        # ========== 核心彻底修复：对非精英子概率列表单独强制归一化 ==========
        non_elite_prob_sum = non_elite_prob.sum()
        if non_elite_prob_sum < 1e-12:  # 极端情况：非精英个体适应度全为0，等概率分配
            non_elite_prob = np.ones_like(non_elite_prob) / len(non_elite_prob)
        else:
            non_elite_prob = non_elite_prob / non_elite_prob_sum  # 强制子概率和=1
        # =====================================================================
    
        # 随机选择：传入归一化后的子概率列表，彻底避免和不为1的错误
        selected_indices = np.random.choice(
            non_elite_indices,
            size=self.pop_size - self.elitism_size,
            p=non_elite_prob,  # 已归一化，和严格为1
            replace=True
        )
        selected = [self.population[i] for i in selected_indices]
    
        # 精英+选择的个体组成新种群
        return elite + selected

    def _crossover(self, parent1: InversionParams, parent2: InversionParams, cross_rate: float) -> Tuple[InversionParams, InversionParams]:
        """
        均匀交叉操作：对每层参数独立生成掩码，决定继承父代1还是父代2
        Args:
            parent1: 父代1
            parent2: 父代2
            cross_rate: 交叉概率
        Returns:
            child1, child2: 两个子代个体
        """
        child1, child2 = {}, {}
        for key in parent1.keys():
            # 生成交叉掩码：大于cross_rate则继承父代1，否则继承父代2
            mask = np.random.rand(self.n_layers) > cross_rate
            child1[key] = np.where(mask, parent1[key], parent2[key])
            child2[key] = np.where(mask, parent2[key], parent1[key])
            # 强制：交叉后仍保证thickness最后1层为0
            if key == "thickness":
                child1[key][-1] = 0.0
                child2[key][-1] = 0.0
        return child1, child2

    def _mutation(self, individual: InversionParams, mut_rate: float) -> InversionParams:
        """
        高斯变异操作：对每层参数添加小幅度高斯噪声，限制在[0,1]归一化区间
        优化：噪声方差随变异率动态调整，提升适配性
        Args:
            individual: 待变异的个体
            mut_rate: 变异概率
        Returns:
            mutated_ind: 变异后的个体
        """
        mutated_ind = {}
        for key in individual.keys():
            mut = individual[key].copy()
            # 生成变异掩码：小于mut_rate则进行变异
            mask = np.random.rand(self.n_layers) < mut_rate
            # ========== 优化点：噪声方差随变异率动态调整 ==========
            noise_scale = 0.1 * mut_rate  # 变异率越低，噪声越小，更稳定
            noise = np.random.normal(loc=0, scale=noise_scale, size=self.n_layers)
            mut[mask] += noise[mask]
            # 将值限制在归一化区间[0,1]
            mut = np.clip(mut, 0.0, 1.0)
            # 强制：变异后仍保证thickness最后1层为0
            if key == "thickness":
                mut[-1] = 0.0
            mutated_ind[key] = mut
        return mutated_ind

    def run(self, observed_curve: DispersionCurve) -> InversionParams:
        """
        运行自适应遗传算法
        Args:
            observed_curve: 观测频散曲线 (频率, 相速度)
        Returns:
            best_params: 全局最优反演参数（反归一化后，含thickness/vs）
        """
        if self.forward_model is None:
            raise RuntimeError("请先通过set_forward_model设置正演函数")
        
        # ========== 修正点4：提前校验观测曲线格式 ==========
        self._validate_observed_curve(observed_curve)
        # 保存观测曲线，用于适应度计算
        self.observed_curve = observed_curve

        # 初始化种群
        self._init_population()
        best_fitness = 0.0
        best_individual = None

        # 迭代进化
        for gen in range(self.max_generations):
            # 1. 计算当前代所有个体的适应度（仅计算一次）
            self._compute_population_fitness()
            
            # 2. 自适应调整交叉率和变异率
            cross_rate, mut_rate = self._adjust_params(gen)
            
            # 3. 选择操作（精英+轮盘赌）
            selected_pop = self._selection()  # 长度=pop_size
            
            # ========== 修正点5：重构交叉变异逻辑，确保执行 ==========
            # 初始化新种群：先保留精英，剩余位置由交叉变异填充
            elite_indices = np.argsort(self.current_fitness)[-self.elitism_size:]
            elite = [self.population[i] for i in elite_indices]
            new_population = elite.copy()  # 初始仅保留精英
            
            # 对非精英的选择个体进行交叉变异
            non_elite_selected = selected_pop[self.elitism_size:]  # 排除精英，避免重复处理
            while len(new_population) < self.pop_size:
                # 随机选择两个父代（从非精英选择个体中选）
                if len(non_elite_selected) < 2:
                    parent1, parent2 = non_elite_selected[0], non_elite_selected[0]  # 兜底：仅1个时自交叉
                else:
                    parent1, parent2 = np.random.choice(non_elite_selected, size=2, replace=False)
                
                # 交叉生成子代
                child1, child2 = self._crossover(parent1, parent2, cross_rate)
                
                # 变异子代
                child1 = self._mutation(child1, mut_rate)
                child2 = self._mutation(child2, mut_rate)
                
                # 添加到新种群
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # 截断到种群大小（避免极端情况超界）
            self.population = new_population[:self.pop_size]
            
            # 4. 更新最优个体
            current_best_idx = np.argmax(self.current_fitness)
            current_best_fitness = self.current_fitness[current_best_idx]
            current_best_ind = self.population[current_best_idx]

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_ind
            
            # 记录适应度历史
            self.fitness_history.append(best_fitness)

            # ========== 修正点6：增强日志信息，便于调试 ==========
            if (gen % 10 == 0 or gen == self.max_generations - 1):
                avg_fitness = np.mean(self.current_fitness)
                print(f"GA迭代{gen+1:3d}/{self.max_generations} | 最优适应度: {best_fitness:.6f} | 平均适应度: {avg_fitness:.6f} | 交叉率: {cross_rate:.3f} | 变异率: {mut_rate:.3f}")

        # 反归一化最优个体，得到物理参数
        best_params = denormalize_params(best_individual, self.param_bounds)
        # 最终校验最优参数
        validate_inversion_params(best_params)
        return best_params