"use strict";(self.webpackChunkbackend_ai_webui_react=self.webpackChunkbackend_ai_webui_react||[]).push([[556],{77758:function(n,e,a){a.d(e,{Z:function(){return o}});var t=a(1413),i=a(36459),l=a(83842),r=(a(4519),a(2556)),o=function(n){var e,a=Object.assign({},((0,i.Z)(n),n));return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)("style",{children:".ant-modal.bai-modal .ant-modal-content {\n  padding: var(--general-modal-content-padding, 0);\n}\n.ant-modal.bai-modal .ant-modal-body {\n  padding: var(--general-modal-body-padding, 0 24px);\n}\n\n.ant-modal.bai-modal .ant-modal-footer {\n  padding: var(--general-modal-footer-padding, 0 20px 24px 20px);\n}\n\n.ant-modal.bai-modal .ant-modal-header {\n  border-bottom: 1px solid rgb(221, 221, 221);\n  border-width: 100%;\n  justify-content: space-between;\n  display: flex;\n  align-items: center;\n}\n\n.ant-modal.bai-modal .ant-modal-content .ant-modal-header,\n.ant-modal.bai-modal .ant-modal-content > button.ant-modal-close {\n  padding: var(--general-modal-header-padding, 10px 20px);\n  height: var(--general-modal-header-height, 69px);\n}\n\n.ant-modal.bai-modal .ant-modal-content > button.ant-modal-close {\n  /* center */\n  top: 0;\n}\n"}),(0,r.jsx)(l.Z,(0,t.Z)({centered:null===(e=a.centered)||void 0===e||e,className:"bai-modal"},a))]})}},61502:function(n,e,a){a.r(e),a.d(e,{default:function(){return L},useShadowRoot:function(){return _},useWebComponentInfo:function(){return F}});var t=a(29439),i=a(74165),l=a(15861),r=a(1413);function o(n,e){var a=(0,r.Z)({},e);return function(n){for(var e,a=/(\w+)\s@\s*(\w+)\s*\(\s*(\w+)\s*:\s*(\$?\w+)\s*\)/g,i=[];null!==(e=a.exec(n));){var l=e,r=(0,t.Z)(l,5),o=r[0],u=r[1],d=r[2],s=r[3],c=r[4];i.push({fieldName:u,directive:d,argumentName:s,argumentValue:c,originFieldStr:o})}return i}(n).forEach((function(t){if("skipOnClient"===t.directive&&"if"===t.argumentName&&(n=!t.argumentValue||!0!==e[t.argumentValue.substring(1)]&&"true"!==t.argumentValue?n.replace(t.originFieldStr,t.originFieldStr.replace(/@\s*(skipOnClient)\s*\(\s*(\w+)\s*:\s*(\$?\w+)\s*\)/,"")):n.replace(t.originFieldStr,""),t.argumentValue.startsWith("$")&&2===n.split(t.argumentValue).length)){var i=t.argumentValue.substring(1),l=new RegExp(".*".concat(i,".*\n"),"g");n=n.replace(l,""),delete a[t.argumentValue.substring(1)]}})),{query:n,variables:a}}var u=a(41011);u.RelayFeatureFlags.ENABLE_RELAY_RESOLVERS=!0;var d=function(){var n=(0,l.Z)((0,i.Z)().mark((function n(e,a){var t,l,r,u,d,s;return(0,i.Z)().wrap((function(n){for(;;)switch(n.prev=n.next){case 0:return r=o(e.text||"",a),u={query:r.query,variables:r.variables},d=null===(t=globalThis.backendaiclient)||void 0===t?void 0:t.newSignedRequest("POST","/admin/gql",u),n.next=5,null===(l=globalThis.backendaiclient)||void 0===l?void 0:l._wrapWithPromise(d,!1,null,1e4,0);case 5:return s=n.sent,n.abrupt("return",s);case 7:case"end":return n.stop()}}),n)})));return function(e,a){return n.apply(this,arguments)}}();var s,c=new u.Environment({network:u.Network.create(d,void 0),store:new u.Store(new u.RecordSource)}),g=a(80382),m=a(27340),f=a(27398),v=a(49883),p=a(66670),b=a(64447),h=a(13881),y=a(4519),k=a(81748),x=a(87112),w=a(16980),S=a(51843),j=a(12674),T=a(2556),Z=y.createContext(null),I=y.createContext(null),_=function(){return y.useContext(I)},F=function(){return y.useContext(Z)},C=new x.QueryClient({defaultOptions:{queries:{suspense:!0,refetchOnWindowFocus:!1,retry:!1}}});b.ZP.use(k.Db).use(h.Z).init({backend:{loadPath:"/resources/i18n/{{lng}}.json"},lng:(null===globalThis||void 0===globalThis||null===(s=globalThis.backendaioptions)||void 0===s?void 0:s.get("current_language"))||"en",fallbackLng:"en",interpolation:{escapeValue:!1}});var E=function(){var n=(0,j.s0)();return(0,y.useLayoutEffect)((function(){var e=function(e){var a=e.detail;n(a,{replace:!0})};return document.addEventListener("react-navigate",e),function(){document.removeEventListener("react-navigate",e)}}),[n]),null},L=function(n){var e=n.children,a=n.value,i=n.styles,l=n.shadowRoot,r=n.dispatchEvent,o=(0,y.useMemo)((function(){return(0,m.Df)()}),[]),u=function(){var n,e=(0,y.useState)(null===globalThis||void 0===globalThis||null===(n=globalThis.backendaioptions)||void 0===n?void 0:n.get("current_language")),a=(0,t.Z)(e,2),i=a[0],l=a[1],r=(0,k.$G)().i18n;return(0,y.useEffect)((function(){setTimeout((function(){return null===r||void 0===r?void 0:r.changeLanguage(i)}),0)}),[]),(0,y.useEffect)((function(){var n=function(n){var e,a;l(null===n||void 0===n||null===(e=n.detail)||void 0===e?void 0:e.lang);var t=(null===n||void 0===n||null===(a=n.detail)||void 0===a?void 0:a.lang)||"en";null===r||void 0===r||r.changeLanguage(t)};return window.addEventListener("langChanged",n),function(){return window.removeEventListener("langChanged",n)}}),[r]),[i]}(),d=(0,t.Z)(u,1)[0],s=(0,g.x)(),b=(0,y.useMemo)((function(){return{value:a,dispatchEvent:r,moveTo:function(n){r("moveTo",{path:n})}}}),[a,r]);return(0,T.jsx)(T.Fragment,{children:c&&(0,T.jsx)(w.RelayEnvironmentProvider,{environment:c,children:(0,T.jsxs)(y.StrictMode,{children:[(0,T.jsxs)("style",{children:[i,".anticon {\n  display: inline-block;\n  color: inherit;\n  font-style: normal;\n  line-height: 0;\n  text-align: center;\n  text-transform: none;\n  vertical-align: -0.125em;\n  text-rendering: optimizeLegibility;\n  -webkit-font-smoothing: antialiased;\n  -moz-osx-font-smoothing: grayscale;\n}\n\n.anticon > * {\n  line-height: 1;\n}\n\n.anticon svg {\n  display: inline-block;\n}\n\n.anticon::before {\n  display: none;\n}\n\n.anticon .anticon-icon {\n  display: block;\n}\n\n.anticon[tabindex] {\n  cursor: pointer;\n}\n\n.anticon-spin::before,\n.anticon-spin {\n  display: inline-block;\n  -webkit-animation: loadingCircle 1s infinite linear;\n  animation: loadingCircle 1s infinite linear;\n}\n\n@-webkit-keyframes loadingCircle {\n  100% {\n    -webkit-transform: rotate(360deg);\n    transform: rotate(360deg);\n  }\n}\n\n@keyframes loadingCircle {\n  100% {\n    -webkit-transform: rotate(360deg);\n    transform: rotate(360deg);\n  }\n}\n\n.ant-table-wrapper {\n  border-radius: 8px 8px 0 0;\n  overflow: hidden;\n}\n"]}),(0,T.jsx)(x.QueryClientProvider,{client:C,children:(0,T.jsx)(I.Provider,{value:l,children:(0,T.jsx)(Z.Provider,{value:b,children:(0,T.jsx)(f.ZP,{getPopupContainer:function(n){return l},locale:"ko"===d?p.Z:v.Z,theme:s,children:(0,T.jsx)(m.V9,{container:l,cache:o,children:(0,T.jsx)(y.Suspense,{fallback:"",children:(0,T.jsxs)(S.VK,{children:[(0,T.jsx)(E,{}),e]})})})})})})})]})})})}},4652:function(n,e,a){a.r(e);var t,i=a(1413),l=a(36459),r=a(87760),o=a(77758),u=a(61502),d=a(12513),s=a(21346),c=a(39883),g=a(82548),m=a(32048),f=a.n(m),v=(a(4519),a(81748)),p=a(87112),b=a(16980),h=a(2556);e.default=function(n){var e,m,y=Object.assign({},((0,l.Z)(n),n)),k=(0,v.$G)().t,x=(0,u.useWebComponentInfo)(),w=x.value,S=x.dispatchEvent;try{m=JSON.parse(w||"")}catch(O){m={open:!1,userEmail:""}}var j,T=m,Z=T.open,I=T.userEmail,_=(0,r.Dj)(),F=(0,p.useQuery)("isManagerSupportingTOTP",(function(){return _.isManagerSupportingTOTP()}),{suspense:!1}),C=F.data,E=F.isLoading;j=(null===_||void 0===_?void 0:_.supports("2FA"))&&C;var L=(0,b.useLazyLoadQuery)(void 0!==t?t:t=a(21960),{email:I,isTOTPSupported:null!==(e=j)&&void 0!==e&&e}).user,K={xxl:1,xl:1,lg:1,md:1,sm:1,xs:1};return(0,h.jsxs)(o.Z,(0,i.Z)((0,i.Z)({open:Z,onCancel:function(){S("cancel",null)},centered:!0,title:k("credential.UserDetail"),footer:[(0,h.jsx)(d.ZP,{type:"primary",onClick:function(){S("cancel",null)},children:k("button.OK")},"ok")]},y),{},{children:[(0,h.jsx)("br",{}),(0,h.jsxs)(s.Z,{size:"small",column:K,title:k("credential.Information"),labelStyle:{width:"50%"},children:[(0,h.jsx)(s.Z.Item,{label:k("credential.UserID"),children:null===L||void 0===L?void 0:L.email}),(0,h.jsx)(s.Z.Item,{label:k("credential.UserName"),children:null===L||void 0===L?void 0:L.username}),(0,h.jsx)(s.Z.Item,{label:k("credential.FullName"),children:null===L||void 0===L?void 0:L.full_name}),(0,h.jsx)(s.Z.Item,{label:k("credential.DescActiveUser"),children:"active"===(null===L||void 0===L?void 0:L.status)?k("button.Yes"):k("button.No")}),(0,h.jsx)(s.Z.Item,{label:k("credential.DescRequirePasswordChange"),children:null!==L&&void 0!==L&&L.need_password_change?k("button.Yes"):k("button.No")}),j&&(0,h.jsx)(s.Z.Item,{label:k("webui.menu.TotpActivated"),children:(0,h.jsx)(c.Z,{spinning:E,children:null!==L&&void 0!==L&&L.totp_activated?k("button.Yes"):k("button.No")})})]}),(0,h.jsx)("br",{}),(0,h.jsxs)(s.Z,{size:"small",column:K,title:k("credential.Association"),labelStyle:{width:"50%"},children:[(0,h.jsx)(s.Z.Item,{label:k("credential.Domain"),children:null===L||void 0===L?void 0:L.domain_name}),(0,h.jsx)(s.Z.Item,{label:k("credential.Role"),children:null===L||void 0===L?void 0:L.role})]}),(0,h.jsx)("br",{}),(0,h.jsx)(s.Z,{title:k("credential.ProjectAndGroup"),labelStyle:{width:"50%"},children:(0,h.jsx)(s.Z.Item,{children:f().map(null===L||void 0===L?void 0:L.groups,(function(n){return(0,h.jsx)(g.Z,{children:null===n||void 0===n?void 0:n.name},null===n||void 0===n?void 0:n.id)}))})})]}))}},21960:function(n,e,a){a.r(e);var t=function(){var n=[{defaultValue:null,kind:"LocalArgument",name:"email"},{defaultValue:null,kind:"LocalArgument",name:"isTOTPSupported"}],e=[{kind:"Variable",name:"email",variableName:"email"}],a={alias:null,args:null,kind:"ScalarField",name:"email",storageKey:null},t={alias:null,args:null,kind:"ScalarField",name:"username",storageKey:null},i={alias:null,args:null,kind:"ScalarField",name:"need_password_change",storageKey:null},l={alias:null,args:null,kind:"ScalarField",name:"full_name",storageKey:null},r={alias:null,args:null,kind:"ScalarField",name:"description",storageKey:null},o={alias:null,args:null,kind:"ScalarField",name:"status",storageKey:null},u={alias:null,args:null,kind:"ScalarField",name:"domain_name",storageKey:null},d={alias:null,args:null,kind:"ScalarField",name:"role",storageKey:null},s={alias:null,args:null,kind:"ScalarField",name:"id",storageKey:null},c={alias:null,args:null,concreteType:"UserGroup",kind:"LinkedField",name:"groups",plural:!0,selections:[s,{alias:null,args:null,kind:"ScalarField",name:"name",storageKey:null}],storageKey:null},g={condition:"isTOTPSupported",kind:"Condition",passingValue:!0,selections:[{alias:null,args:null,kind:"ScalarField",name:"totp_activated",storageKey:null}]};return{fragment:{argumentDefinitions:n,kind:"Fragment",metadata:null,name:"UserInfoModalQuery",selections:[{alias:null,args:e,concreteType:"User",kind:"LinkedField",name:"user",plural:!1,selections:[a,t,i,l,r,o,u,d,c,g],storageKey:null}],type:"Queries",abstractKey:null},kind:"Request",operation:{argumentDefinitions:n,kind:"Operation",name:"UserInfoModalQuery",selections:[{alias:null,args:e,concreteType:"User",kind:"LinkedField",name:"user",plural:!1,selections:[a,t,i,l,r,o,u,d,c,g,s],storageKey:null}]},params:{cacheID:"0df215d9d53a960adda4cd628fa40661",id:null,metadata:{},name:"UserInfoModalQuery",operationKind:"query",text:"query UserInfoModalQuery(\n  $email: String\n  $isTOTPSupported: Boolean!\n) {\n  user(email: $email) {\n    email\n    username\n    need_password_change\n    full_name\n    description\n    status\n    domain_name\n    role\n    groups {\n      id\n      name\n    }\n    totp_activated @include(if: $isTOTPSupported)\n    id\n  }\n}\n"}}}();t.hash="ebbace65870261723ee661def143e3e8",e.default=t},87760:function(n,e,a){a.d(e,{Dj:function(){return s},Kr:function(){return r},M:function(){return d},dS:function(){return c},qh:function(){return u},tQ:function(){return o}});var t=a(29439),i=a(4519),l=a(87112),r=function(n){return function(n){var e=(0,i.useState)(n||(new Date).toISOString()),a=(0,t.Z)(e,2),l=a[0],r=a[1];return[l,function(n){r(n||(new Date).toISOString())}]}(n)},o=function(){return s()._config.domainName},u=function(){var n=s(),e=(0,i.useState)({name:n.current_group,id:n.groupIds[n.current_group]}),a=(0,t.Z)(e,2),l=a[0],r=a[1];return(0,i.useEffect)((function(){var e=function(e){var a=e.detail;r({name:a,id:n.groupIds[a]})};return document.addEventListener("backend-ai-group-changed",e),function(){document.removeEventListener("backend-ai-group-changed",e)}})),l},d=function(n){var e=n.api_endpoint;return(0,i.useMemo)((function(){var n=new globalThis.BackendAIClientConfig("","",e,"SESSION");return new globalThis.BackendAIClient(n,"Backend.AI Console.")}),[e])},s=function(){return(0,l.useQuery)({queryKey:"backendai-client-for-suspense",queryFn:function(){return new Promise((function(n){if("undefined"!==typeof globalThis.backendaiclient&&null!==globalThis.backendaiclient&&!1!==globalThis.backendaiclient.ready)return n(globalThis.backendaiclient);document.addEventListener("backend-ai-connected",(function e(){n(globalThis.backendaiclient),document.removeEventListener("backend-ai-connected",e)}))}))},retry:!1,suspense:!0}).data},c=function(){var n=(0,l.useQuery)({queryKey:"backendai-metadata-for-suspense",queryFn:function(){return fetch("resources/image_metadata.json").then((function(n){return n.json()})).then((function(n){return n}))},suspense:!0,retry:!1}).data,e=function(n){if(!n)return{key:"",tags:[]};var e=n.split("/"),a=(e[2]||e[1]).split(":"),i=(0,t.Z)(a,2);return{key:i[0],tags:i[1].split("-")}};return[n,{getImageAliasName:function(a){var t=e(a).key;return(null===n||void 0===n?void 0:n.imageInfo[t].name)||t},getImageIcon:function(a){var t,i,l=arguments.length>1&&void 0!==arguments[1]?arguments[1]:"resources/icons/";if(!a)return"default.png";var r=e(a).key;return l+(void 0!==(null===n||void 0===n||null===(t=n.imageInfo[r])||void 0===t?void 0:t.icon)?null===n||void 0===n||null===(i=n.imageInfo[r])||void 0===i?void 0:i.icon:"default.png")},getImageTags:function(n){},getBaseVersion:function(n){return e(n).tags[0]},getBaseImage:function(n){return e(n).tags[1]},getImageMeta:e}]}}}]);
//# sourceMappingURL=556.fbaa1103.chunk.js.map