// Dash's debounced number input can retain the value from before a store sync
// in its Enter/blur handlers. Commit the DOM value so editing back to that
// earlier value works too. Keep updates on Enter/blur rather than every key.
(function () {
    function commitWeight(event) {
        const input = event.target;
        if (!(input instanceof HTMLInputElement) ||
            !input.closest('.mlpeg-weight-input') ||
            !input.id || !input.checkValidity() || input.value === '') {
            return;
        }
        const value = input.valueAsNumber;
        if (Number.isFinite(value) && window.dash_clientside?.set_props) {
            window.dash_clientside.set_props(input.id, {value: value});
        }
    }
    document.addEventListener('keyup', function (event) {
        if (event.key === 'Enter') commitWeight(event);
    });
    document.addEventListener('blur', commitWeight, true);
})();
