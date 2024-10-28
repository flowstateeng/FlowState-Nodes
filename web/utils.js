import {ComfyWidgets} from '../../scripts/widgets.js'

export function createTextWidget(app, node, widgetName, styles = {}) {
    const widget = ComfyWidgets['STRING'](node, widgetName, ['STRING', {multiline: true}], app).widget;
    widget.inputEl.readOnly = true;
    Object.assign(widget.inputEl.style, styles);
    return widget;
}

// export function createListWidget(app, node, widgetName, options = [], styles = {}) {
//     const widget = ComfyWidgets['LIST'](node, widgetName, ['LIST', { options }], app).widget;
//     Object.assign(widget.inputEl.style, styles);
//     return widget;
// }


