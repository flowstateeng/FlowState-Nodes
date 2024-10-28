import { app } from '../../scripts/app.js'
import { createTextWidget, createListWidget } from './utils.js'

app.registerExtension({
    name: 'FlowStateNodes.FlowStatePromptOutput',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === 'FlowStatePromptOutput') {
            const onNodeCreated = nodeType.prototype.onNodeCreated

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments)
                const text = createTextWidget(app, this, 'text')
                const nodeWidth = this.size[0]
                const nodeHeight = this.size[1]
                this.setSize([nodeWidth, nodeHeight * 3])
                return result
            }

            const onExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments)
                this.widgets.find(obj => obj.name === 'text').value = message.text
            }
        }
    },
})

// app.registerExtension({
//     name: 'FlowStateNodes.FlowStateUnifiedStyler',
//     async beforeRegisterNodeDef(nodeType, nodeData, app) {
//         if (nodeData.name === 'FlowStateUnifiedStyler') {
//             console.log(`\n\nNODE DATA: `, nodeData , `\n\n`)
//             const onNodeCreated = nodeType.prototype.onNodeCreated

//             nodeType.prototype.onNodeCreated = function () {
//                 const result = onNodeCreated?.apply(this, arguments)

//                 const options = ['Option 1', 'Option 2', 'Option 3']
//                 const list = createListWidget(app, this, 'list', options)

//                 const nodeWidth = this.size[0]
//                 const nodeHeight = this.size[1]
//                 this.setSize([nodeWidth, nodeHeight * 3])
//                 return result
//             }

//             const onExecuted = nodeType.prototype.onExecuted
//             nodeType.prototype.onExecuted = function (message) {
//                 onExecuted?.apply(this, arguments)
//                 const listWidget = this.widgets.find(obj => obj.name === 'list')

//                 // Assign value(s) to the list
//                 listWidget.value = message.selectedOption  // Set the selected item based on message contents
//             }
//         }
//     },
// })


