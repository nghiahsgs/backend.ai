import{r as e,ad as t,aJ as a,aK as i,aL as r,aM as s,q as n,Z as l,p as d,T as o,J as u,P as h,i as p,$ as c,al as g,a8 as m,a9 as v,a2 as _}from"./backend-ai-webui-DHPXkWFV.js";
/**
 * @license
 * Copyright (c) 2017 - 2024 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */e("vaadin-text-field",t,{moduleId:"lumo-text-field-styles"});
/**
 * @license
 * Copyright (c) 2021 - 2024 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */
const f=e=>class extends(a(e)){static get properties(){return{autocomplete:{type:String},autocorrect:{type:String},autocapitalize:{type:String,reflectToAttribute:!0}}}static get delegateAttrs(){return[...super.delegateAttrs,"autocapitalize","autocomplete","autocorrect"]}get __data(){return this.__dataValue||{}}set __data(e){this.__dataValue=e}_inputElementChanged(e){super._inputElementChanged(e),e&&(e.value&&e.value!==this.value&&(console.warn(`Please define value on the <${this.localName}> component!`),e.value=""),this.value&&(e.value=this.value))}_setFocused(e){super._setFocused(e),!e&&document.hasFocus()&&this.validate()}_onInput(e){super._onInput(e),this.invalid&&this.validate()}_valueChanged(e,t){super._valueChanged(e,t),void 0!==t&&this.invalid&&this.validate()}}
/**
 * @license
 * Copyright (c) 2021 - 2024 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */,x=e=>class extends(f(e)){static get properties(){return{maxlength:{type:Number},minlength:{type:Number},pattern:{type:String}}}static get delegateAttrs(){return[...super.delegateAttrs,"maxlength","minlength","pattern"]}static get constraints(){return[...super.constraints,"maxlength","minlength","pattern"]}constructor(){super(),this._setType("text")}get clearElement(){return this.$.clearButton}ready(){super.ready(),this.addController(new i(this,(e=>{this._setInputElement(e),this._setFocusElement(e),this.stateTarget=e,this.ariaTarget=e}))),this.addController(new r(this.inputElement,this._labelController))}}
/**
 * @license
 * Copyright (c) 2017 - 2024 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */;e("vaadin-text-field",s,{moduleId:"vaadin-text-field-styles"});class y extends(x(o(u(h)))){static get is(){return"vaadin-text-field"}static get template(){return n`
      <div class="vaadin-field-container">
        <div part="label">
          <slot name="label"></slot>
          <span part="required-indicator" aria-hidden="true" on-click="focus"></span>
        </div>

        <vaadin-input-container
          part="input-field"
          readonly="[[readonly]]"
          disabled="[[disabled]]"
          invalid="[[invalid]]"
          theme$="[[_theme]]"
        >
          <slot name="prefix" slot="prefix"></slot>
          <slot name="input"></slot>
          <slot name="suffix" slot="suffix"></slot>
          <div id="clearButton" part="clear-button" slot="suffix" aria-hidden="true"></div>
        </vaadin-input-container>

        <div part="helper-text">
          <slot name="helper"></slot>
        </div>

        <div part="error-message">
          <slot name="error-message"></slot>
        </div>
      </div>
      <slot name="tooltip"></slot>
    `}static get properties(){return{maxlength:{type:Number},minlength:{type:Number}}}ready(){super.ready(),this._tooltipController=new l(this),this._tooltipController.setPosition("top"),this._tooltipController.setAriaTarget(this.inputElement),this.addController(this._tooltipController)}}d(y),
/**
 * @license
 * Copyright (c) 2016 - 2024 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */
e("vaadin-grid-filter",p`
    :host {
      display: inline-flex;
      max-width: 100%;
    }

    ::slotted(*) {
      width: 100%;
      box-sizing: border-box;
    }
  `,{moduleId:"vaadin-grid-filter-styles"});const b=e=>class extends(c(e)){static get properties(){return{path:{type:String,sync:!0},value:{type:String,notify:!0,sync:!0},_textField:{type:Object,sync:!0}}}static get observers(){return["_filterChanged(path, value, _textField)"]}ready(){super.ready(),this._filterController=new g(this,"","vaadin-text-field",{initializer:e=>{e.addEventListener("input",(e=>{this.value=e.target.value})),this._textField=e}}),this.addController(this._filterController)}_filterChanged(e,t,a){void 0!==e&&void 0!==t&&a&&(a.value=t,this._debouncerFilterChanged=m.debounce(this._debouncerFilterChanged,v.after(200),(()=>{this.dispatchEvent(new CustomEvent("filter-changed",{bubbles:!0}))})))}focus(){this._textField&&this._textField.focus()}}
/**
 * @license
 * Copyright (c) 2016 - 2024 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */;class C extends(b(o(h))){static get template(){return n`<slot></slot>`}static get is(){return"vaadin-grid-filter"}}d(C);
/**
 * @license
 * Copyright (c) 2016 - 2024 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */
const E=e=>class extends e{static get properties(){return{path:{type:String,sync:!0},header:{type:String,sync:!0}}}static get observers(){return["_onHeaderRendererOrBindingChanged(_headerRenderer, _headerCell, path, header)"]}_defaultHeaderRenderer(e,t){let a=e.firstElementChild,i=a?a.firstElementChild:void 0;a||(a=document.createElement("vaadin-grid-filter"),i=document.createElement("vaadin-text-field"),i.setAttribute("theme","small"),i.setAttribute("style","max-width: 100%;"),i.setAttribute("focus-target",""),a.appendChild(i),e.appendChild(a)),a.path=this.path,i.label=this.__getHeader(this.header,this.path)}_computeHeaderRenderer(){return this._defaultHeaderRenderer}__getHeader(e,t){return e||(t?this._generateHeader(t):void 0)}}
/**
 * @license
 * Copyright (c) 2016 - 2024 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */;class F extends(E(_)){static get is(){return"vaadin-grid-filter-column"}}d(F);
