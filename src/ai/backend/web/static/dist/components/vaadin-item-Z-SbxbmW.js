import{q as t,p as s,T as a,D as e,P as i}from"./backend-ai-webui-DHPXkWFV.js";import{I as r}from"./vaadin-item-mixin-DuqXhN3A.js";
/**
 * @license
 * Copyright (c) 2017 - 2024 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */class n extends(r(a(e(i)))){static get template(){return t`
      <style>
        :host {
          display: inline-block;
        }

        :host([hidden]) {
          display: none !important;
        }
      </style>
      <span part="checkmark" aria-hidden="true"></span>
      <div part="content">
        <slot></slot>
      </div>
    `}static get is(){return"vaadin-item"}constructor(){super(),this.value,this.label}ready(){super.ready(),this.setAttribute("role","option")}}s(n);
