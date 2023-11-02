import{G as e}from"./vaadin-grid-8c354e58.js";
/**
 * @license
 * Copyright (c) 2016 - 2023 Vaadin Ltd.
 * This program is available under Apache License Version 2.0, available at https://vaadin.com/license/
 */class t extends e{static get is(){return"vaadin-grid-selection-column"}static get properties(){return{width:{type:String,value:"58px"},flexGrow:{type:Number,value:0},selectAll:{type:Boolean,value:!1,notify:!0},autoSelect:{type:Boolean,value:!1},__indeterminate:Boolean,__previousActiveItem:Object,__selectAllHidden:Boolean}}static get observers(){return["__onSelectAllChanged(selectAll)","_onHeaderRendererOrBindingChanged(_headerRenderer, _headerCell, path, header, selectAll, __indeterminate, __selectAllHidden)"]}constructor(){super(),this.__boundOnActiveItemChanged=this.__onActiveItemChanged.bind(this),this.__boundOnDataProviderChanged=this.__onDataProviderChanged.bind(this),this.__boundOnSelectedItemsChanged=this.__onSelectedItemsChanged.bind(this)}disconnectedCallback(){this._grid.removeEventListener("active-item-changed",this.__boundOnActiveItemChanged),this._grid.removeEventListener("data-provider-changed",this.__boundOnDataProviderChanged),this._grid.removeEventListener("filter-changed",this.__boundOnSelectedItemsChanged),this._grid.removeEventListener("selected-items-changed",this.__boundOnSelectedItemsChanged),super.disconnectedCallback()}connectedCallback(){super.connectedCallback(),this._grid&&(this._grid.addEventListener("active-item-changed",this.__boundOnActiveItemChanged),this._grid.addEventListener("data-provider-changed",this.__boundOnDataProviderChanged),this._grid.addEventListener("filter-changed",this.__boundOnSelectedItemsChanged),this._grid.addEventListener("selected-items-changed",this.__boundOnSelectedItemsChanged))}_defaultHeaderRenderer(e,t){let i=e.firstElementChild;i||(i=document.createElement("vaadin-checkbox"),i.setAttribute("aria-label","Select All"),i.classList.add("vaadin-grid-select-all-checkbox"),i.addEventListener("checked-changed",this.__onSelectAllCheckedChanged.bind(this)),e.appendChild(i));const d=this.__isChecked(this.selectAll,this.__indeterminate);i.__rendererChecked=d,i.checked=d,i.hidden=this.__selectAllHidden,i.indeterminate=this.__indeterminate}_defaultRenderer(e,t,{item:i,selected:d}){let r=e.firstElementChild;r||(r=document.createElement("vaadin-checkbox"),r.setAttribute("aria-label","Select Row"),r.addEventListener("checked-changed",this.__onSelectRowCheckedChanged.bind(this)),e.appendChild(r)),r.__item=i,r.__rendererChecked=d,r.checked=d}__onSelectAllChanged(e){void 0!==e&&this._grid&&(this.__selectAllInitialized?this._selectAllChangeLock||(e&&this.__hasArrayDataProvider()?this.__withFilteredItemsArray((e=>{this._grid.selectedItems=e})):this._grid.selectedItems=[]):this.__selectAllInitialized=!0)}__arrayContains(e,t){return Array.isArray(e)&&Array.isArray(t)&&t.every((t=>e.includes(t)))}__onSelectAllCheckedChanged(e){e.target.checked!==e.target.__rendererChecked&&(this.selectAll=this.__indeterminate||e.target.checked)}__onSelectRowCheckedChanged(e){e.target.checked!==e.target.__rendererChecked&&(e.target.checked?this._grid.selectItem(e.target.__item):this._grid.deselectItem(e.target.__item))}__isChecked(e,t){return t||e}__onActiveItemChanged(e){const t=e.detail.value;if(this.autoSelect){const e=t||this.__previousActiveItem;e&&this._grid._toggleItem(e)}this.__previousActiveItem=t}__hasArrayDataProvider(){return Array.isArray(this._grid.items)&&!!this._grid.dataProvider}__onSelectedItemsChanged(){this._selectAllChangeLock=!0,this.__hasArrayDataProvider()&&this.__withFilteredItemsArray((e=>{this._grid.selectedItems.length?this.__arrayContains(this._grid.selectedItems,e)?(this.selectAll=!0,this.__indeterminate=!1):(this.selectAll=!1,this.__indeterminate=!0):(this.selectAll=!1,this.__indeterminate=!1)})),this._selectAllChangeLock=!1}__onDataProviderChanged(){this.__selectAllHidden=!Array.isArray(this._grid.items)}__withFilteredItemsArray(e){const t={page:0,pageSize:1/0,sortOrders:[],filters:this._grid._mapFilters()};this._grid.dataProvider(t,(t=>e(t)))}}customElements.define(t.is,t);
