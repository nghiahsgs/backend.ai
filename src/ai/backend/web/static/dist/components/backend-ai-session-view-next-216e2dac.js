import{_ as s,e as i,B as t,c as a,I as r,a as e,i as o,x as n,j as m,k as c}from"./backend-ai-webui-f8c1c4e3.js";import"./backend-ai-session-launcher-c83bd1eb.js";import"./backend-ai-session-view-43be12d1.js";import"./lablup-codemirror-8ce3a351.js";import"./lablup-progress-bar-4824463c.js";import"./slider-e5e5a450.js";import"./mwc-check-list-item-c89e37b0.js";import"./media-query-controller-55feaab5.js";import"./dir-utils-d56795bb.js";import"./vaadin-grid-7658779f.js";import"./vaadin-grid-filter-column-8da99835.js";import"./vaadin-grid-selection-column-1f26e01a.js";import"./json_to_csv-35c9e191.js";import"./backend-ai-resource-monitor-574a8003.js";import"./mwc-switch-f67f3f80.js";import"./backend-ai-list-status-0c01dfa8.js";import"./lablup-grid-sort-filter-column-64966434.js";import"./vaadin-grid-sort-column-bfdd8fb2.js";import"./vaadin-iconset-6dbb4538.js";import"./lablup-activity-panel-ee675e01.js";import"./mwc-formfield-46184177.js";import"./mwc-tab-bar-d124c910.js";let p=class extends t{static get styles(){return[a,r,e,o``]}async _viewStateChanged(s){await this.updateComplete}render(){return n`
      <backend-ai-react-session-list
        @moveTo="${s=>{const i=s.detail.path;globalThis.history.pushState({},"",i),m.dispatch(c(decodeURIComponent(i),{}))}}"
      ></backend-ai-react-session-list>
    `}};p=s([i("backend-ai-session-view-next")],p);
