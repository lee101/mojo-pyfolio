"""Portfolio return kernels exported through a stable C ABI."""

from max.algorithm import sync_parallelize
from std.math import isnan, log, sqrt
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime MOMENT_WORKERS = 8
comptime PARALLEL_MOMENT_ELEMENTS = 1_048_576
comptime CUM_RETURN_WORKERS = 8
comptime PARALLEL_CUM_RETURN_ELEMENTS = 1_048_576


@always_inline
def ptr(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


@export("mpf_cum_returns")
def mpf_cum_returns(
    source_address: Int,
    destination_address: Int,
    rows: Int,
    columns: Int,
    starting_value: Float64,
) abi("C"):
    comptime W = simdwidthof[DType.float64]()
    var source = ptr(source_address)
    var destination = ptr(destination_address)
    if columns == 1:
        var wealth = 1.0
        var index = 0
        if starting_value == 0.0:
            while index + W <= rows:
                var values = source.unsafe_load[width=W](offset=index)
                var cumulative = SIMD[DType.float64, W](0.0)
                comptime for lane in range(W):
                    var value = values[lane]
                    if not isnan(value):
                        wealth *= 1.0 + value
                    cumulative[lane] = wealth - 1.0
                destination.unsafe_store(index, cumulative)
                index += W
            while index < rows:
                var value = source[unsafe_offset=index]
                if not isnan(value):
                    wealth *= 1.0 + value
                destination[unsafe_offset=index] = wealth - 1.0
                index += 1
        else:
            while index + W <= rows:
                var values = source.unsafe_load[width=W](offset=index)
                var cumulative = SIMD[DType.float64, W](0.0)
                comptime for lane in range(W):
                    var value = values[lane]
                    if not isnan(value):
                        wealth *= 1.0 + value
                    cumulative[lane] = wealth * starting_value
                destination.unsafe_store(index, cumulative)
                index += W
            while index < rows:
                var value = source[unsafe_offset=index]
                if not isnan(value):
                    wealth *= 1.0 + value
                destination[unsafe_offset=index] = wealth * starting_value
                index += 1
        return

    var vector_groups = columns // W
    var workers = min(num_physical_cores(), CUM_RETURN_WORKERS)
    workers = min(workers, vector_groups)
    if rows * columns < PARALLEL_CUM_RETURN_ELEMENTS:
        workers = 1
    workers = max(workers, 1)

    @__parameter
    def process_vector_groups(worker: Int):
        var source = ptr(source_address)
        var destination = ptr(destination_address)
        var first_group = worker * vector_groups // workers
        var last_group = (worker + 1) * vector_groups // workers
        for group in range(first_group, last_group):
            var column = group * W
            var wealth = SIMD[DType.float64, W](1.0)
            for row in range(rows):
                var offset = row * columns + column
                var values = source.unsafe_load[width=W](offset=offset)
                var valid = values.eq(values)
                wealth *= valid.select(
                    values + 1.0, SIMD[DType.float64, W](1.0)
                )
                destination.unsafe_store(
                    offset,
                    wealth - 1.0
                    if starting_value == 0.0
                    else wealth * starting_value,
                )

    if vector_groups > 0:
        if workers > 1:
            sync_parallelize[process_vector_groups](workers)
        else:
            process_vector_groups(0)

    for column in range(vector_groups * W, columns):
        var wealth = 1.0
        for row in range(rows):
            var value = source[unsafe_offset=row * columns + column]
            if not isnan(value):
                wealth *= 1.0 + value
            destination[unsafe_offset=row * columns + column] = (
                wealth - 1.0 if starting_value == 0.0 else wealth * starting_value
            )


@export("mpf_max_drawdown")
def mpf_max_drawdown(
    source_address: Int, length: Int, result_address: Int
) abi("C"):
    var source = ptr(source_address)
    var result = ptr(result_address)
    var wealth = 100.0
    var peak = 100.0
    var maximum = 0.0
    for index in range(length):
        var value = source[unsafe_offset=index]
        if not isnan(value):
            wealth *= 1.0 + value
        if wealth > peak:
            peak = wealth
        var drawdown = (wealth - peak) / peak
        if drawdown < maximum:
            maximum = drawdown
    result[unsafe_offset=0] = maximum


@export("mpf_drawdown_series")
def mpf_drawdown_series(
    source_address: Int, destination_address: Int, length: Int
) abi("C"):
    var source = ptr(source_address)
    var destination = ptr(destination_address)
    var wealth = 100.0
    var peak = 100.0
    for index in range(length):
        var value = source[unsafe_offset=index]
        if not isnan(value):
            wealth *= 1.0 + value
        if wealth > peak:
            peak = wealth
        destination[unsafe_offset=index] = (wealth - peak) / peak


@export("mpf_return_summary")
def mpf_return_summary(
    source_address: Int,
    length: Int,
    required_return: Float64,
    omega_offset: Float64,
    result_address: Int,
) abi("C"):
    """Fused nan-aware moments, wealth, drawdown, and log-path regression."""
    var source = ptr(source_address)
    var result = ptr(result_address)
    for index in range(11):
        result[unsafe_offset=index] = 0.0

    var wealth = 100.0
    var peak = 100.0
    var max_drawdown = 0.0
    var cumulative_log = 0.0
    var valid_index = 0
    result[unsafe_offset=6] = 1.0

    for index in range(length):
        var value = source[unsafe_offset=index]
        if isnan(value):
            continue

        result[unsafe_offset=0] += 1.0
        result[unsafe_offset=1] += value
        result[unsafe_offset=2] += value * value

        var downside = value - required_return
        if downside < 0.0:
            result[unsafe_offset=3] += downside * downside

        var omega_value = value - omega_offset
        if omega_value > 0.0:
            result[unsafe_offset=4] += omega_value
        elif omega_value < 0.0:
            result[unsafe_offset=5] -= omega_value

        result[unsafe_offset=6] *= 1.0 + value
        wealth *= 1.0 + value
        if wealth > peak:
            peak = wealth
        var drawdown = (wealth - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown

        cumulative_log += log(1.0 + value)
        var x = Float64(valid_index)
        result[unsafe_offset=8] += cumulative_log
        result[unsafe_offset=9] += cumulative_log * cumulative_log
        result[unsafe_offset=10] += x * cumulative_log
        valid_index += 1

    result[unsafe_offset=7] = max_drawdown


@export("mpf_central_moments")
def mpf_central_moments(
    source_address: Int, length: Int, result_address: Int
) abi("C"):
    """Return the mean and biased central moments two through four."""
    var workers = min(num_physical_cores(), MOMENT_WORKERS)
    if length < PARALLEL_MOMENT_ELEMENTS:
        workers = 1
    workers = max(workers, 1)

    @__parameter
    def sum_chunk(worker: Int):
        comptime W = simdwidthof[DType.float64]()
        var source = ptr(source_address)
        var partial = ptr(result_address)
        var vectors = length // W
        var start = (worker * vectors // workers) * W
        var end = ((worker + 1) * vectors // workers) * W
        if worker == workers - 1:
            end = length
        var vector_sum = SIMD[DType.float64, W](0.0)
        var index = start
        while index + W <= end:
            vector_sum += source.unsafe_load[width=W](offset=index)
            index += W
        var total = vector_sum.reduce_add()
        while index < end:
            total += source[unsafe_offset=index]
            index += 1
        partial[unsafe_offset=worker] = total

    for worker in range(workers):
        sum_chunk(worker)

    var result = ptr(result_address)
    var total = 0.0
    for worker in range(workers):
        total += result[unsafe_offset=worker]
    var mean = total / Float64(length)

    @__parameter
    def moment_chunk(worker: Int):
        comptime W = simdwidthof[DType.float64]()
        var source = ptr(source_address)
        var partial = ptr(result_address)
        var vectors = length // W
        var start = (worker * vectors // workers) * W
        var end = ((worker + 1) * vectors // workers) * W
        if worker == workers - 1:
            end = length
        var second = SIMD[DType.float64, W](0.0)
        var third = SIMD[DType.float64, W](0.0)
        var fourth = SIMD[DType.float64, W](0.0)
        var index = start
        while index + W <= end:
            var delta = source.unsafe_load[width=W](offset=index) - mean
            var squared = delta * delta
            second += squared
            third += squared * delta
            fourth += squared * squared
            index += W
        var second_total = second.reduce_add()
        var third_total = third.reduce_add()
        var fourth_total = fourth.reduce_add()
        while index < end:
            var delta = source[unsafe_offset=index] - mean
            var squared = delta * delta
            second_total += squared
            third_total += squared * delta
            fourth_total += squared * squared
            index += 1
        partial[unsafe_offset=worker * 3] = second_total
        partial[unsafe_offset=worker * 3 + 1] = third_total
        partial[unsafe_offset=worker * 3 + 2] = fourth_total

    for worker in range(workers):
        moment_chunk(worker)

    var second_total = 0.0
    var third_total = 0.0
    var fourth_total = 0.0
    for worker in range(workers):
        second_total += result[unsafe_offset=worker * 3]
        third_total += result[unsafe_offset=worker * 3 + 1]
        fourth_total += result[unsafe_offset=worker * 3 + 2]
    result[unsafe_offset=0] = mean
    result[unsafe_offset=1] = second_total / Float64(length)
    result[unsafe_offset=2] = third_total / Float64(length)
    result[unsafe_offset=3] = fourth_total / Float64(length)


@export("mpf_factor_summary")
def mpf_factor_summary(
    returns_address: Int,
    factor_address: Int,
    length: Int,
    result_address: Int,
) abi("C"):
    var returns = ptr(returns_address)
    var factor = ptr(factor_address)
    var result = ptr(result_address)
    for index in range(5):
        result[unsafe_offset=index] = 0.0
    for index in range(length):
        var y = returns[unsafe_offset=index]
        var x = factor[unsafe_offset=index]
        if isnan(y) or isnan(x):
            continue
        result[unsafe_offset=0] += 1.0
        result[unsafe_offset=1] += y
        result[unsafe_offset=2] += x
        result[unsafe_offset=3] += x * x
        result[unsafe_offset=4] += x * y


@export("mpf_rolling_stats")
def mpf_rolling_stats(
    source_address: Int,
    volatility_address: Int,
    sharpe_address: Int,
    length: Int,
    window: Int,
    annualization: Float64,
    mode: Int,
) abi("C"):
    var source = ptr(source_address)
    var volatility = ptr(volatility_address)
    var sharpe = ptr(sharpe_address)
    var total = 0.0
    var squares = 0.0
    var valid = 0
    var annual_root = sqrt(annualization)

    for index in range(length):
        var entering = source[unsafe_offset=index]
        if not isnan(entering):
            total += entering
            squares += entering * entering
            valid += 1
        if index >= window:
            var leaving = source[unsafe_offset=index - window]
            if not isnan(leaving):
                total -= leaving
                squares -= leaving * leaving
                valid -= 1
        if index >= window - 1 and valid == window:
            var mean = total / Float64(window)
            var variance = (
                squares - total * total / Float64(window)
            ) / Float64(window - 1)
            if variance < 0.0 and variance > -1.0e-24:
                variance = 0.0
            var deviation = sqrt(variance)
            if mode != 2:
                volatility[unsafe_offset=index] = deviation * annual_root
            if mode != 1:
                sharpe[unsafe_offset=index] = mean / deviation * annual_root


@export("mpf_rolling_beta")
def mpf_rolling_beta(
    returns_address: Int,
    factor_address: Int,
    destination_address: Int,
    length: Int,
    rolling_window: Int,
) abi("C"):
    """Match pyfolio's inclusive label slice: each result uses window + 1 rows."""
    var returns = ptr(returns_address)
    var factor = ptr(factor_address)
    var destination = ptr(destination_address)
    var count = 0
    var sum_x = 0.0
    var sum_y = 0.0
    var sum_xx = 0.0
    var sum_xy = 0.0
    var full_window = rolling_window + 1

    for index in range(length):
        var y = returns[unsafe_offset=index]
        var x = factor[unsafe_offset=index]
        if not isnan(y) and not isnan(x):
            count += 1
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_xy += x * y

        if index >= full_window:
            var old_y = returns[unsafe_offset=index - full_window]
            var old_x = factor[unsafe_offset=index - full_window]
            if not isnan(old_y) and not isnan(old_x):
                count -= 1
                sum_x -= old_x
                sum_y -= old_y
                sum_xx -= old_x * old_x
                sum_xy -= old_x * old_y

        if index >= rolling_window and count > 0:
            var n = Float64(count)
            var covariance = sum_xy - sum_x * sum_y / n
            var factor_variance = sum_xx - sum_x * sum_x / n
            if factor_variance >= 1.0e-30:
                destination[unsafe_offset=index] = covariance / factor_variance


@export("mpf_summarize_paths")
def mpf_summarize_paths(
    samples_address: Int,
    mean_address: Int,
    deviation_address: Int,
    samples_count: Int,
    days: Int,
    starting_value: Float64,
) abi("C"):
    var samples = ptr(samples_address)
    var means = ptr(mean_address)
    var deviations = ptr(deviation_address)
    for day in range(days):
        means[unsafe_offset=day] = 0.0
        deviations[unsafe_offset=day] = 0.0

    for sample in range(samples_count):
        var wealth = starting_value
        for day in range(days):
            var value = samples[unsafe_offset=sample * days + day]
            if not isnan(value):
                wealth *= 1.0 + value
            means[unsafe_offset=day] += wealth
            deviations[unsafe_offset=day] += wealth * wealth

    for day in range(days):
        var mean = means[unsafe_offset=day] / Float64(samples_count)
        var variance = deviations[unsafe_offset=day] / Float64(samples_count) - mean * mean
        means[unsafe_offset=day] = mean
        deviations[unsafe_offset=day] = sqrt(max(0.0, variance))
