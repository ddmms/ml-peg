/* Record settings locally before asynchronous score callbacks run. Store writes
 * are patches so independent control groups cannot replace each other's edits.
 * Each editor owns both directions of its controls, avoiding callback echoes. */
(function () {
    "use strict";
    window.dash_clientside = window.dash_clientside || {};

    const nu = () => window.dash_clientside.no_update;
    const changed = () => new Set(
        (window.dash_clientside.callback_context.triggered || []).map(x => x.prop_id));
    // Debounced controls report Enter/blur before publishing their value.
    // A newly mounted control has neither, so its default cannot overwrite
    // a persisted setting. Store-to-input restoration does not count as an edit.
    const committed = (args, count, index) =>
        (args[count + 1 + index * 2] || 0) > 0 ||
        (args[count + 2 + index * 2] || 0) > 0;
    const assignment = (location, value) => ({operation: "Assign", location,
        params: {value}});
    const patch = operations => operations.length
        ? {__dash_patch_update: "__dash_patch_update", operations} : nu();

    /* Python round(x, digits) rounds the exact binary float, with ties to even.
     * Scaling a Number then using Math.round introduces a second rounding step.
     * Integer arithmetic retains the binary value until decimal conversion. */
    function roundEven(value, digits) {
        if (!Number.isFinite(value) || Math.abs(value) >= 2 ** 52) return value;
        const view = new DataView(new ArrayBuffer(8));
        view.setFloat64(0, Math.abs(value));
        const bits = view.getBigUint64(0);
        const exponent = Number((bits >> 52n) & 2047n);
        const mantissa = (bits & ((1n << 52n) - 1n)) + (exponent ? 1n << 52n : 0n);
        const shift = exponent ? exponent - 1023 - 52 : -1074;
        const scale = 10n ** BigInt(digits);
        const numerator = mantissa * scale;
        let integer;
        if (shift >= 0) {
            integer = numerator << BigInt(shift);
        } else {
            const denominator = 1n << BigInt(-shift);
            integer = numerator / denominator;
            const remainder = numerator % denominator;
            if (2n * remainder > denominator ||
                (2n * remainder === denominator && integer % 2n === 1n)) integer++;
        }
        const sign = value < 0 || Object.is(value, -0) ? "-" : "";
        return Number(sign + (integer / scale).toString() + "." +
            (integer % scale).toString().padStart(digits, "0"));
    }

    function correctBounds(field, good, bad, defaults, step, digits) {
        if (defaults.good > defaults.bad && good <= bad) {
            if (field === "good") bad = good - step;
            else good = bad + step;
        } else if (defaults.good < defaults.bad && good >= bad) {
            if (field === "good") bad = good + step;
            else good = bad - step;
        }
        return [roundEven(good, digits), roundEven(bad, digits)];
    }

    function weights(config, stored, ...args) {
        const values = args.slice(0, config.columns.length);
        const clicks = args[config.columns.length];
        const triggers = changed();
        const next = {...(stored || {})};
        const operations = [];
        if (triggers.has(config.reset + ".n_clicks") && clicks) {
            for (const key of new Set([...Object.keys(next), ...Object.keys(config.defaults)])) {
                next[key] = config.defaults[key] ?? 1;
                operations.push(assignment([key], next[key]));
            }
        } else {
            config.columns.forEach((key, i) => {
                if (!triggers.has(config.inputs[i] + ".value") ||
                    !committed(args, config.columns.length, i)) return;
                const value = values[i];
                if (typeof value !== "number" || !Number.isFinite(value)) return;
                if (next[key] !== value) {
                    next[key] = value;
                    operations.push(assignment([key], value));
                }
            });
        }
        return [patch(operations), ...config.columns.map((key, i) =>
            (next[key] ?? config.defaults[key] ?? 1) !== values[i]
                ? (next[key] ?? config.defaults[key] ?? 1) : nu())];
    }

    function thresholds(config, stored, ...args) {
        const values = args.slice(0, config.metrics.length * 2);
        const clicks = args[config.metrics.length * 2];
        const triggers = changed();
        const next = {...(stored || {})};
        const operations = [];
        if (triggers.has(config.reset + ".n_clicks") && clicks) {
            for (const [metric, entry] of Object.entries(config.defaults)) {
                next[metric] = {...entry};
                operations.push(assignment([metric], next[metric]));
            }
        } else {
            config.metrics.forEach((metric, i) => {
                const defaults = config.defaults[metric];
                const original = next[metric];
                if (!defaults || !original) return;
                let good = Number(original.good), bad = Number(original.bad);
                if (!Number.isFinite(good) || !Number.isFinite(bad)) return;
                let edited = false;
                for (const [offset, field] of ["good", "bad"].entries()) {
                    const index = i * 2 + offset;
                    if (!triggers.has(config.inputs[index] + ".value") ||
                        !committed(args, values.length, index)) continue;
                    const value = values[index];
                    if (typeof value !== "number" || !Number.isFinite(value)) continue;
                    [good, bad] = correctBounds(field,
                        field === "good" ? value : good,
                        field === "bad" ? value : bad,
                        defaults, config.step, config.digits);
                    edited = true;
                }
                if (edited && (good !== original.good || bad !== original.bad)) {
                    next[metric] = {...original, good, bad};
                    operations.push(assignment([metric], next[metric]));
                }
            });
        }
        return [patch(operations), ...config.metrics.flatMap((metric, i) => {
            const entry = next[metric];
            return ["good", "bad"].map((field, offset) => {
                const value = entry && entry[field] != null ? Number(entry[field]) : NaN;
                return Number.isFinite(value) && value !== values[i * 2 + offset]
                    ? value : nu();
            });
        })];
    }

    function presets(config, applyClicks, resetClicks, pending) {
        const triggers = changed();
        if (triggers.has("weight-preset-reset.n_clicks") && resetClicks) {
            return [patch(Object.entries(config.defaults).map(
                ([key, value]) => assignment([key], value))), []];
        }
        if (!triggers.has("weight-preset-apply.n_clicks") ||
            !applyClicks || !pending?.length) return [nu(), nu()];
        const weights = Object.fromEntries(config.columns.map(key => [key, 0]));
        for (const preset of pending) {
            for (const column of config.columns) {
                const value = preset === "Random" ? roundEven(Math.random(), 2)
                    : Number(config.presets[preset].some(title => column === title + " Score"));
                weights[column] = Math.max(weights[column], value);
            }
        }
        return [patch(Object.entries(weights).map(
            ([key, value]) => assignment([key], value))), nu()];
    }

    window.dash_clientside.settingsEditors = {
        weights, thresholds, presets, roundEven, correctBounds
    };
})();
