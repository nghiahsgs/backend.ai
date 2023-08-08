import{aA as r,a3 as e,_ as t,a4 as s,a6 as a,aB as o,aC as i}from"./backend-ai-webui-661d9e43.js";const n=285;let f=class extends r{render(){return a`
			<svg focusable="false" preserveAspectRatio="xMidYMid meet" viewBox="0 0 100 100">
				${this.bufferPerc>0?o`<circle id="buffer" cx="50%" cy="50%" r="45" fill="transparent" style="stroke-dashoffset: ${n-n*this.bufferPerc}px"></circle>`:""}
				<circle
					id="progress"
					cx="50%"
					cy="50%"
					r="45"
					fill="transparent"
					style="${this.mode===i.DETERMINATE?`stroke-dashoffset: ${n-n*this.progressPerc}px;`:""}"
				></circle>
			</svg>
		`}};f.styles=[...r.styles,e(":host{width:var(--progress-spinner-size,3rem);height:var(--progress-spinner-size,3rem);contain:strict;position:relative;display:inline-block}svg{width:100%;height:100%;position:absolute;transform:rotate(-90deg);top:0;left:0;transform-origin:center center;overflow:visible}#buffer,#progress{stroke-width:var(--progress-spinner-stroke-width,10%);stroke-dasharray:285px;transition-property:stroke;transform-origin:center}#buffer{stroke:var(--progress-spinner-buffer-color,hsla(var(--primary-500,var(--primary-hue,224),var(--primary-saturation,47%),var(--primary-lightness,38%)),.2))}#progress{stroke:var(--progress-spinner-color,hsl(var(--primary-500,var(--primary-hue,224),var(--primary-saturation,47%),var(--primary-lightness,38%))))}:host([mode=indeterminate]) svg{animation:rotate var(--progress-spinner-indeterminate-container-duration,2s) linear infinite}:host([mode=indeterminate]) #progress{animation-duration:var(--progress-spinner-indeterminate-progress-duration,4s);animation-timing-function:var(--progress-spinner-indeterminate-timing-function,var(--transition-timing-function-standard-curve,cubic-bezier(.4,0,.2,1)));animation-iteration-count:infinite;animation-name:indeterminate}#buffer,:host([mode=determinate]) #progress{transition:var(--progress-spinner-determinate-progress-transition,stroke-dashoffset var(--transition-timing-function-linear,linear))}@keyframes rotate{0%{transform:rotate(0)}to{transform:rotate(1turn)}}@keyframes indeterminate{0%{stroke-dashoffset:268px;transform:rotate(0)}12.5%{stroke-dashoffset:56px;transform:rotate(0)}12.5001%{stroke-dashoffset:56px;transform:rotateX(180deg) rotate(72.5deg)}25%{stroke-dashoffset:268px;transform:rotateX(180deg) rotate(72.5deg)}25.0001%{stroke-dashoffset:268px;transform:rotate(270deg)}37.5%{stroke-dashoffset:56px;transform:rotate(270deg)}37.5001%{stroke-dashoffset:56px;transform:rotateX(180deg) rotate(161.5deg)}50%{stroke-dashoffset:268px;transform:rotateX(180deg) rotate(161.5deg)}50.0001%{stroke-dashoffset:268px;transform:rotate(180deg)}62.5%{stroke-dashoffset:56px;transform:rotate(180deg)}62.5001%{stroke-dashoffset:56px;transform:rotateX(180deg) rotate(251.5deg)}75%{stroke-dashoffset:268px;transform:rotateX(180deg) rotate(251.5deg)}75.0001%{stroke-dashoffset:268px;transform:rotate(90deg)}87.5%{stroke-dashoffset:56px;transform:rotate(90deg)}87.5001%{stroke-dashoffset:56px;transform:rotateX(180deg) rotate(341.5deg)}to{stroke-dashoffset:268px;transform:rotateX(180deg) rotate(341.5deg)}}")],f=t([s("wl-progress-spinner")],f);
