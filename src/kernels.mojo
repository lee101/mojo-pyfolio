"""Portfolio return kernels exported through a stable C ABI."""

from std.algorithm import parallelize
from std.math import isnan, log, sqrt
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime MOMENT_WORKERS = 8
comptime PARALLEL_MOMENT_ELEMENTS = 1_048_576


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
    var source = ptr(source_address)
    var destination = ptr(destination_address)
    if columns == 1:
        var wealth = 1.0
        var index = 0
        if starting_value == 0.0:
            while index < rows:
                var value = source[index]
                if not isnan(value):
                    wealth *= 1.0 + value
                destination[index] = wealth - 1.0
                index += 1
        else:
            while index < rows:
                var value = source[index]
                if not isnan(value):
                    wealth *= 1.0 + value
                destination[index] = wealth * starting_value
                index += 1
        return

    for column in range(columns):
        var wealth = 1.0
        for row in range(rows):
            var value = source[row * columns + column]
            if not isnan(value):
                wealth *= 1.0 + value
            destination[row * columns + column] = (
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
        var value = source[index]
        if not isnan(value):
            wealth *= 1.0 + value
        if wealth > peak:
            peak = wealth
        var drawdown = (wealth - peak) / peak
        if drawdown < maximum:
            maximum = drawdown
    result[0] = maximum


@export("mpf_drawdown_series")
def mpf_drawdown_series(
    source_address: Int, destination_address: Int, length: Int
) abi("C"):
    var source = ptr(source_address)
    var destination = ptr(destination_address)
    var wealth = 100.0
    var peak = 100.0
    for index in range(length):
        var value = source[index]
        if not isnan(value):
            wealth *= 1.0 + value
        if wealth > peak:
            peak = wealth
        destination[index] = (wealth - peak) / peak


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
        result[index] = 0.0

    var wealth = 100.0
    var peak = 100.0
    var max_drawdown = 0.0
    var cumulative_log = 0.0
    var valid_index = 0
    result[6] = 1.0

    for index in range(length):
        var value = source[index]
        if isnan(value):
            continue

        result[0] += 1.0
        result[1] += value
        result[2] += value * value

        var downside = value - required_return
        if downside < 0.0:
            result[3] += downside * downside

        var omega_value = value - omega_offset
        if omega_value > 0.0:
            result[4] += omega_value
        elif omega_value < 0.0:
            result[5] -= omega_value

        result[6] *= 1.0 + value
        wealth *= 1.0 + value
        if wealth > peak:
            peak = wealth
        var drawdown = (wealth - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown

        cumulative_log += log(1.0 + value)
        var x = Float64(valid_index)
        result[8] += cumulative_log
        result[9] += cumulative_log * cumulative_log
        result[10] += x * cumulative_log
        valid_index += 1

    result[7] = max_drawdown


@export("mpf_central_moments")
def mpf_central_moments(
    source_address: Int, length: Int, result_address: Int
) abi("C"):
    """Return the mean and biased central moments two through four."""
    var workers = min(num_physical_cores(), MOMENT_WORKERS)
    if length < PARALLEL_MOMENT_ELEMENTS:
        workers = 1
    workers = max(workers, 1)

    @parameter
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
            vector_sum += source.load[width=W](index)
            index += W
        var total = vector_sum.reduce_add()
        while index < end:
            total += source[index]
            index += 1
        partial[worker] = total

    if workers > 1:
        parallelize[sum_chunk](workers, workers)
    else:
        sum_chunk(0)

    var result = ptr(result_address)
    var total = 0.0
    for worker in range(workers):
        total += result[worker]
    var mean = total / Float64(length)

    @parameter
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
            var delta = source.load[width=W](index) - mean
            var squared = delta * delta
            second += squared
            third += squared * delta
            fourth += squared * squared
            index += W
        var second_total = second.reduce_add()
        var third_total = third.reduce_add()
        var fourth_total = fourth.reduce_add()
        while index < end:
            var delta = source[index] - mean
            var squared = delta * delta
            second_total += squared
            third_total += squared * delta
            fourth_total += squared * squared
            index += 1
        partial[worker * 3] = second_total
        partial[worker * 3 + 1] = third_total
        partial[worker * 3 + 2] = fourth_total

    if workers > 1:
        parallelize[moment_chunk](workers, workers)
    else:
        moment_chunk(0)

    var second_total = 0.0
    var third_total = 0.0
    var fourth_total = 0.0
    for worker in range(workers):
        second_total += result[worker * 3]
        third_total += result[worker * 3 + 1]
        fourth_total += result[worker * 3 + 2]
    result[0] = mean
    result[1] = second_total / Float64(length)
    result[2] = third_total / Float64(length)
    result[3] = fourth_total / Float64(length)


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
        result[index] = 0.0
    for index in range(length):
        var y = returns[index]
        var x = factor[index]
        if isnan(y) or isnan(x):
            continue
        result[0] += 1.0
        result[1] += y
        result[2] += x
        result[3] += x * x
        result[4] += x * y


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
        var entering = source[index]
        if not isnan(entering):
            total += entering
            squares += entering * entering
            valid += 1
        if index >= window:
            var leaving = source[index - window]
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
                volatility[index] = deviation * annual_root
            if mode != 1:
                sharpe[index] = mean / deviation * annual_root


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
        var y = returns[index]
        var x = factor[index]
        if not isnan(y) and not isnan(x):
            count += 1
            sum_x += x
            sum_y += y
            sum_xx += x * x
            sum_xy += x * y

        if index >= full_window:
            var old_y = returns[index - full_window]
            var old_x = factor[index - full_window]
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
                destination[index] = covariance / factor_variance


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
        means[day] = 0.0
        deviations[day] = 0.0

    for sample in range(samples_count):
        var wealth = starting_value
        for day in range(days):
            var value = samples[sample * days + day]
            if not isnan(value):
                wealth *= 1.0 + value
            means[day] += wealth
            deviations[day] += wealth * wealth

    for day in range(days):
        var mean = means[day] / Float64(samples_count)
        var variance = deviations[day] / Float64(samples_count) - mean * mean
        means[day] = mean
        deviations[day] = sqrt(max(0.0, variance))
