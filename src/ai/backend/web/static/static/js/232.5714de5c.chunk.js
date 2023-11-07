"use strict";(self.webpackChunkbackend_ai_webui_react=self.webpackChunkbackend_ai_webui_react||[]).push([[232,502,959],{77758:function(e,n,i){i.d(n,{Z:function(){return o}});var l=i(1413),t=i(36459),r=i(65113),a=(i(4519),i(2556)),o=function(e){var n,i=Object.assign({},((0,t.Z)(e),e));return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)("style",{children:".ant-modal.bai-modal .ant-modal-content {\n  padding: var(--general-modal-content-padding, 0);\n}\n.ant-modal.bai-modal .ant-modal-body {\n  padding: var(--general-modal-body-padding, 0 24px);\n}\n\n.ant-modal.bai-modal .ant-modal-footer {\n  padding: var(--general-modal-footer-padding, 0 20px 24px 20px);\n}\n\n.ant-modal.bai-modal .ant-modal-header {\n  border-bottom: 1px solid rgb(221, 221, 221);\n  border-width: 100%;\n  justify-content: space-between;\n  display: flex;\n  align-items: center;\n}\n\n.ant-modal.bai-modal .ant-modal-content .ant-modal-header,\n.ant-modal.bai-modal .ant-modal-content > button.ant-modal-close {\n  padding: var(--general-modal-header-padding, 10px 20px);\n  height: var(--general-modal-header-height, 69px);\n}\n\n.ant-modal.bai-modal .ant-modal-content > button.ant-modal-close {\n  /* center */\n  top: 0;\n}\n"}),(0,a.jsx)(r.Z,(0,l.Z)({centered:null===(n=i.centered)||void 0===n||n,className:"bai-modal"},i))]})}},5959:function(e,n,i){i.r(n);var l=i(28499),t=(i(4519),i(2556));n.default=function(e){var n=e.text,i=e.children;return(0,t.jsx)(l.Z.Text,{copyable:!0,code:!0,children:n||i})}},61502:function(e,n,i){i.r(n),i.d(n,{default:function(){return Q},useShadowRoot:function(){return $},useWebComponentInfo:function(){return A}});var l=i(29439),t=i(74165),r=i(15861),a=i(1413);function o(e,n){var i=(0,a.Z)({},n);return function(e){for(var n,i=/(\w+)\s@\s*(\w+)\s*\(\s*(\w+)\s*:\s*(\$?\w+)\s*\)/g,t=[];null!==(n=i.exec(e));){var r=n,a=(0,l.Z)(r,5),o=a[0],s=a[1],d=a[2],u=a[3],c=a[4];t.push({fieldName:s,directive:d,argumentName:u,argumentValue:c,originFieldStr:o})}return t}(e).forEach((function(l){if("skipOnClient"===l.directive&&"if"===l.argumentName&&(e=!l.argumentValue||!0!==n[l.argumentValue.substring(1)]&&"true"!==l.argumentValue?e.replace(l.originFieldStr,l.originFieldStr.replace(/@\s*(skipOnClient)\s*\(\s*(\w+)\s*:\s*(\$?\w+)\s*\)/,"")):e.replace(l.originFieldStr,""),l.argumentValue.startsWith("$")&&2===e.split(l.argumentValue).length)){var t=l.argumentValue.substring(1),r=new RegExp(".*".concat(t,".*\n"),"g");e=e.replace(r,""),delete i[l.argumentValue.substring(1)]}})),{query:e,variables:i}}var s=i(41011);s.RelayFeatureFlags.ENABLE_RELAY_RESOLVERS=!0;var d=function(){var e=(0,r.Z)((0,t.Z)().mark((function e(n,i){var l,r,a,s,d,u;return(0,t.Z)().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return a=o(n.text||"",i),s={query:a.query,variables:a.variables},d=null===(l=globalThis.backendaiclient)||void 0===l?void 0:l.newSignedRequest("POST","/admin/gql",s),e.next=5,null===(r=globalThis.backendaiclient)||void 0===r?void 0:r._wrapWithPromise(d,!1,null,1e4,0).catch((function(e){}));case 5:if(e.t0=e.sent,e.t0){e.next=8;break}e.t0={};case 8:return u=e.t0,e.abrupt("return",u);case 10:case"end":return e.stop()}}),e)})));return function(n,i){return e.apply(this,arguments)}}();var u,c=new s.Environment({network:s.Network.create(d,void 0),store:new s.Store(new s.RecordSource)}),m=i(80382),g=i(27340),v=i(41239),p=i(47226),f=i(49883),h=i(66670),x=i(99517),y=i.n(x),S=(i(24989),i(79354)),k=i.n(S),b=i(79748),Z=i.n(b),j=i(51714),_=i.n(j),F=i(63540),w=i.n(F),T=i(18272),E=i.n(T),L=i(9666),C=i.n(L),K=i(64447),I=i(13881),P=i(4519),R=i(81748),O=i(87112),G=i(16980),M=i(51843),N=i(12674),D=i(55144),V=i(73181),U=i(2556);y().extend(C()),y().extend(k()),y().extend(Z()),y().extend(_()),y().extend(E()),y().extend(w());var q=P.createContext(null),z=P.createContext(null),$=function(){return P.useContext(z)},A=function(){return P.useContext(q)},B=new O.QueryClient({defaultOptions:{queries:{suspense:!0,refetchOnWindowFocus:!1,retry:!1}}});K.ZP.use(R.Db).use(I.Z).init({backend:{loadPath:"/resources/i18n/{{lng}}.json"},lng:(null===globalThis||void 0===globalThis||null===(u=globalThis.backendaioptions)||void 0===u?void 0:u.get("current_language"))||"en",fallbackLng:"en",interpolation:{escapeValue:!1},react:{transSupportBasicHtmlNodes:!0,transKeepBasicHtmlNodesFor:["br","strong","span","code","p"]}});var H=function(){var e=(0,N.s0)();return(0,P.useLayoutEffect)((function(){var n=function(n){var i=n.detail;e(i,{replace:!0})};return document.addEventListener("react-navigate",n),function(){document.removeEventListener("react-navigate",n)}}),[e]),null},Q=function(e){var n=e.children,i=e.value,t=e.styles,r=e.shadowRoot,a=e.dispatchEvent,o=(0,P.useMemo)((function(){return(0,g.Df)()}),[]),s=function(){var e,n=(0,P.useState)(null===globalThis||void 0===globalThis||null===(e=globalThis.backendaioptions)||void 0===e?void 0:e.get("current_language")),i=(0,l.Z)(n,2),t=i[0],r=i[1],a=(0,R.$G)().i18n;return(0,P.useEffect)((function(){setTimeout((function(){return null===a||void 0===a?void 0:a.changeLanguage(t)}),0),y().locale(t)}),[]),(0,P.useEffect)((function(){var e=function(e){var n,i;r(null===e||void 0===e||null===(n=e.detail)||void 0===n?void 0:n.lang);var l=(null===e||void 0===e||null===(i=e.detail)||void 0===i?void 0:i.lang)||"en";null===a||void 0===a||a.changeLanguage(l),y().locale(l)};return window.addEventListener("langChanged",e),function(){return window.removeEventListener("langChanged",e)}}),[a]),[t]}(),d=(0,l.Z)(s,1)[0],u=(0,m.x)(),x=(0,P.useMemo)((function(){return{value:i,dispatchEvent:a,moveTo:function(e,n){a("moveTo",{path:e,params:n})}}}),[i,a]);return(0,U.jsx)(U.Fragment,{children:c&&(0,U.jsx)(G.RelayEnvironmentProvider,{environment:c,children:(0,U.jsxs)(P.StrictMode,{children:[(0,U.jsxs)("style",{children:[t,".anticon {\n  display: inline-block;\n  color: inherit;\n  font-style: normal;\n  line-height: 0;\n  text-align: center;\n  text-transform: none;\n  vertical-align: -0.125em;\n  text-rendering: optimizeLegibility;\n  -webkit-font-smoothing: antialiased;\n  -moz-osx-font-smoothing: grayscale;\n}\n\n.anticon > * {\n  line-height: 1;\n}\n\n.anticon svg {\n  display: inline-block;\n}\n\n.anticon::before {\n  display: none;\n}\n\n.anticon .anticon-icon {\n  display: block;\n}\n\n.anticon[tabindex] {\n  cursor: pointer;\n}\n\n.anticon-spin::before,\n.anticon-spin {\n  display: inline-block;\n  -webkit-animation: loadingCircle 1s infinite linear;\n  animation: loadingCircle 1s infinite linear;\n}\n\n@-webkit-keyframes loadingCircle {\n  100% {\n    -webkit-transform: rotate(360deg);\n    transform: rotate(360deg);\n  }\n}\n\n@keyframes loadingCircle {\n  100% {\n    -webkit-transform: rotate(360deg);\n    transform: rotate(360deg);\n  }\n}\n\n/* fix: fixed column shadow display outside of a table wrapper */\n.ant-table-wrapper {\n  border-radius: 8px 8px 0 0;\n  overflow: hidden;\n}\n\n/* fix: the tooltip does not appear in the `<Form.Item tooltip={'something'}` when the popup container is a parent node of the trigger node. */\n.ant-form-item-label {\n  overflow: unset !important;\n}\n"]}),(0,U.jsx)(O.QueryClientProvider,{client:B,children:(0,U.jsx)(z.Provider,{value:r,children:(0,U.jsx)(q.Provider,{value:x,children:(0,U.jsx)(v.ZP,{getPopupContainer:function(e){return(null===e||void 0===e?void 0:e.parentNode)||r},locale:"ko"===d?h.Z:f.Z,theme:u,children:(0,U.jsx)(p.Z,{children:(0,U.jsx)(g.V9,{container:r,cache:o,children:(0,U.jsx)(P.Suspense,{fallback:"",children:(0,U.jsx)(M.VK,{children:(0,U.jsxs)(D.QueryParamProvider,{adapter:V.Q,options:{},children:[(0,U.jsx)(H,{}),n]})})})})})})})})})]})})})}},96451:function(e,n,i){var l,t=i(82548),r=(i(4519),i(16980)),a=i(2556);n.Z=function(e){var n,o=e.endpointFrgmt,s=(0,r.useFragment)(void 0!==l?l:l=i(58393),o),d="default";switch(null===s||void 0===s||null===(n=s.status)||void 0===n?void 0:n.toUpperCase()){case"RUNNING":case"HEALTHY":d="success"}return(0,a.jsx)(t.Z,{color:d,children:null===s||void 0===s?void 0:s.status})}},14001:function(e,n,i){var l=i(1413),t=i(29439),r=i(87760),a=i(4519),o=i(2556),s=function(e,n){var i=e.image,a=e.style,s=void 0===a?{}:a,d=arguments.length>2&&void 0!==arguments[2]?arguments[2]:"",u=(0,r.dS)(),c=(0,t.Z)(u,2)[1].getImageIcon;return(0,o.jsx)("img",{src:c(i),style:(0,l.Z)({width:"1em",height:"1em"},s),alt:d})};n.Z=a.memo(s)},76587:function(e,n,i){var l,t=i(1413),r=i(29439),a=i(44925),o=i(43255),s=i(87760),d=i(50164),u=i(77758),c=i(99277),m=i(44036),g=i(57054),v=i(66653),p=(i(4519),i(81748)),f=i(16980),h=i(2556),x=["onRequestClose","endpointFrgmt"];n.Z=function(e){var n=e.onRequestClose,y=e.endpointFrgmt,S=(0,a.Z)(e,x),k=m.Z.useToken().token,b=(0,s.Dj)(),Z=(0,p.$G)().t,j=g.Z.useForm(),_=(0,r.Z)(j,1)[0],F=(0,f.useFragment)(void 0!==l?l:l=i(56326),y),w=(0,d.Y)({mutationFn:function(e){var n={to:e.desired_session_count};return(0,o.Lc)({method:"POST",url:"/services/".concat(null===F||void 0===F?void 0:F.endpoint_id,"/scale"),body:n,client:b})}});return(0,h.jsx)(u.Z,(0,t.Z)((0,t.Z)({style:{zIndex:1e4},destroyOnClose:!0,onOk:function(e){_.validateFields().then((function(e){w.mutate(e,{onSuccess:function(){console.log("service updated"),n(!0)},onError:function(e){console.log(e)}})})).catch((function(e){console.log(e)}))},onCancel:function(){n()},okButtonProps:{loading:w.isLoading},title:Z("modelService.EditModelService")},S),{},{children:(0,h.jsx)(c.Z,{direction:"row",align:"stretch",justify:"around",children:(0,h.jsx)(g.Z,{form:_,preserve:!1,validateTrigger:["onChange","onBlur"],initialValues:{desired_session_count:null===F||void 0===F?void 0:F.desired_session_count},style:{marginBottom:k.marginLG,marginTop:k.margin},children:(0,h.jsx)(g.Z.Item,{name:"desired_session_count",label:Z("modelService.DesiredSessionCount"),rules:[{pattern:/^[0-9]+$/,message:Z("modelService.OnlyAllowsNonNegativeIntegers")}],children:(0,h.jsx)(v.Z,{type:"number",min:0})})})})}))}},96732:function(e,n,i){i.d(n,{Ec:function(){return g}});var l=i(44925),t=i(1413),r=i(43255),a=i(99277),o=i(44036),s=i(28499),d=i(227),u=(i(4519),i(81748)),c=i(2556),m=["type","size","showIcon","showUnit","showTooltip"],g={"cuda.device":"GPU","cuda.shares":"FGPU","rocm.device":"GPU","tpu.device":"TPU","ipu.device":"IPU","atom.device":"ATOM","warboy.device":"Warboy"},v=function(e){var n=e.size,i=void 0===n?16:n,l=e.children;return(0,c.jsx)("mwc-icon",{style:{"--mdc-icon-size":"".concat(i+2,"px"),width:i,height:i},children:l})},p=function(e){var n,i,r,a=e.type,o=e.size,s=void 0===o?16:o,g=(e.showIcon,e.showUnit,e.showTooltip),p=void 0===g||g,f=(0,l.Z)(e,m),h=(0,u.$G)().t,x={cpu:[(0,c.jsx)(v,{size:s,children:"developer_board"}),h("session.core")],mem:[(0,c.jsx)(v,{size:s,children:"memory"}),"GiB"],"cuda.device":["/resources/icons/file_type_cuda.svg","GPU"],"cuda.shares":["/resources/icons/file_type_cuda.svg","FGPU"],"rocm.device":["/resources/icons/ROCm.png","GPU"],"tpu.device":[(0,c.jsx)(v,{size:s,children:"view_module"}),"TPU"],"ipu.device":[(0,c.jsx)(v,{size:s,children:"view_module"}),"IPU"],"atom.device":["/resources/icons/rebel.svg","ATOM"],"warboy.device":["/resources/icons/furiosa.svg","Warboy"]};return(0,c.jsx)(d.Z,{title:p?"".concat(a," (").concat(x[a][1],")"):void 0,children:"string"===typeof(null===(n=x[a])||void 0===n?void 0:n[0])?(0,c.jsx)("img",(0,t.Z)((0,t.Z)({},f),{},{style:(0,t.Z)({height:s},f.style||{}),src:(null===(i=x[a])||void 0===i?void 0:i[0])||"",alt:a})):(0,c.jsx)("div",{style:{width:16,height:16},children:(null===(r=x[a])||void 0===r?void 0:r[0])||a})})};n.ZP=function(e){var n=e.type,i=e.value,l=e.extra,d=e.opts,m=(0,u.$G)().t,v=o.Z.useToken().token,f=(0,t.Z)({cpu:m("session.core"),mem:"GiB"},g);return(0,c.jsxs)(a.Z,{direction:"row",gap:"xxs",children:[(0,c.jsx)(p,{type:n}),(0,c.jsx)(s.Z.Text,{children:"GiB"===f[n]?(0,r.PZ)(i+"b","g",2).numberFixed:"FGPU"===f[n]?parseFloat(i).toFixed(2):i}),(0,c.jsx)(s.Z.Text,{type:"secondary",children:f[n]}),"mem"===n&&(null===d||void 0===d?void 0:d.shmem)&&(0,c.jsxs)(s.Z.Text,{type:"secondary",style:{fontSize:v.fontSizeSM},children:["(SHM: ",(0,r.PZ)(d.shmem+"b","g",2).numberFixed,"GiB)"]}),l]})}},58393:function(e,n,i){i.r(n);var l={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"EndpointStatusTagFragment",selections:[{alias:null,args:null,kind:"ScalarField",name:"id",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"status",storageKey:null}],type:"Endpoint",abstractKey:null,hash:"3b31efa50b55edddcb210b59003dc479"};n.default=l},56326:function(e,n,i){i.r(n);var l={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"ModelServiceSettingModal_endpoint",selections:[{alias:null,args:null,kind:"ScalarField",name:"endpoint_id",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"desired_session_count",storageKey:null}],type:"Endpoint",abstractKey:null,hash:"881f18324b27eba6ff0fcfb83ae241d2"};n.default=l},90146:function(e,n,i){i.r(n);var l={argumentDefinitions:[],kind:"Fragment",metadata:null,name:"ServingRouteErrorModalFragment",selections:[{alias:null,args:null,kind:"ScalarField",name:"session_id",storageKey:null},{alias:null,args:null,concreteType:"InferenceSessionErrorInfo",kind:"LinkedField",name:"errors",plural:!0,selections:[{alias:null,args:null,kind:"ScalarField",name:"repr",storageKey:null}],storageKey:null}],type:"InferenceSessionError",abstractKey:null,hash:"a1003e0f75387e665f4407909eea5ff6"};n.default=l},23232:function(e,n,i){i.r(n),i.d(n,{default:function(){return te}});var l,t,r=i(29439),a=i(5959),o=i(96451),s=i(1413),d=i(44925),u=i(43255),c=i(87760),m=i(50164),g=i(77758),v=i(99277),p=i(57054),f=i(88464),h=i(52176),x=i(66957),y=i(99517),S=i.n(y),k=i(4519),b=i(81748),Z=i(2556),j=["onRequestClose","onCancel","endpoint_id"],_=function(e){var n=e.onRequestClose,i=(e.onCancel,e.endpoint_id),l=(0,d.Z)(e,j),t=(0,b.$G)().t,a=(0,c.Dj)(),o=p.Z.useForm(),y=(0,r.Z)(o,1)[0],k=(0,m.Y)({mutationFn:function(e){var n={valid_until:e.valid_until};return(0,u.Lc)({method:"POST",url:"/services/".concat(i,"/token"),body:n,client:a})}});return(0,Z.jsx)(g.Z,(0,s.Z)((0,s.Z)({},l),{},{destroyOnClose:!0,onOk:function(e){y.validateFields().then((function(e){var i=e.datetime.unix();k.mutate({valid_until:i},{onSuccess:function(){f.ZP.success(t("modelService.TokenGenerated")),n(!0)},onError:function(e){var n;null!==e&&void 0!==e&&null!==(n=e.message)&&void 0!==n&&n.includes("valid_until is older than now")?f.ZP.error(t("modelService.TokenExpiredDateError")):(f.ZP.error(t("modelService.TokenGenerationFailed")),console.log(e))}})}))},onCancel:function(){n()},okText:t("modelService.Generate"),confirmLoading:k.isLoading,centered:!0,title:t("modelService.GenerateNewToken"),children:(0,Z.jsx)(p.Z,{preserve:!1,labelCol:{span:10},initialValues:{datetime:S()().add(24,"hour")},validateTrigger:["onChange","onBlur"],style:{maxWidth:500},form:y,children:(0,Z.jsxs)(v.Z,{direction:"column",gap:"sm",align:"stretch",children:[(0,Z.jsx)(h.Z,{type:"info",showIcon:!0,message:t("modelService.TokenExpiredDateHelp")}),(0,Z.jsx)(v.Z,{direction:"row",align:"stretch",justify:"around",children:(0,Z.jsx)(p.Z.Item,{name:"datetime",label:t("modelService.ExpiredDate"),rules:[{type:"object",required:!0,message:t("modelService.PleaseSelectTime")},function(){return{validator:function(e,n){return n.isAfter(S()())?Promise.resolve():Promise.reject(new Error(t("modelService.TokenExpiredDateError")))}}}],children:(0,Z.jsx)(x.Z,{showTime:!0,format:"YYYY-MM-DD HH:mm:ss",style:{width:200}})})})]})})}))},F=i(14001),w=i(76587),T=i(96732),E=i(92171),L=i(93448),C=i(16980),K=["onRequestClose","onCancel","inferenceSessionErrorFrgmt"],I=function(e){var n=e.onRequestClose,t=(e.onCancel,e.inferenceSessionErrorFrgmt),r=(0,d.Z)(e,K),o=(0,b.$G)().t,u=(0,C.useFragment)(void 0!==l?l:l=i(90146),t);return(0,Z.jsx)(g.Z,(0,s.Z)((0,s.Z)({centered:!0,title:o("modelService.ServingRouteErrorModalTitle"),onCancel:function(){n()},footer:[(0,Z.jsx)(E.ZP,{onClick:function(){n()},children:o("button.Close")})]},r),{},{children:(0,Z.jsxs)(L.Z,{bordered:!0,column:{xxl:1,xl:1,lg:1,md:1,sm:1,xs:1},labelStyle:{minWidth:100},style:{marginTop:20},children:[(0,Z.jsx)(L.Z.Item,{label:o("modelService.SessionId"),children:(0,Z.jsx)(a.default,{children:null===u||void 0===u?void 0:u.session_id})}),(0,Z.jsx)(L.Z.Item,{label:o("dialog.error.Error"),children:null===u||void 0===u?void 0:u.errors[0].repr})]})}))},P=i(61502),R=i(14644),O=i(28499),G=function(e){var n=e.uuid,i=e.clickable,l=(0,c.qh)(),t=(0,u.y3)(),r=(0,P.useWebComponentInfo)().moveTo,a=(0,m.h)({queryKey:["VFolderSelectQuery"],queryFn:function(){return t({method:"GET",url:"/folders?group_id=".concat(l.id)})},staleTime:1e3,suspense:!0}).data,o=null===a||void 0===a?void 0:a.find((function(e){return e.id===n.replaceAll("-","")}));return o&&(i?(0,Z.jsxs)(O.Z.Link,{onClick:function(){r("/data",{folder:o.name})},children:[(0,Z.jsx)(R.Z,{})," ",o.name]}):(0,Z.jsxs)("div",{children:[(0,Z.jsx)(R.Z,{})," ",o.name]}))},M=i(43971),N=i(72842),D=i(43596),V=i(83861),U=i(31662),q=i(32064),z=i(20558),$=i(56713),A=i(44036),B=i(60284),H=i(227),Q=i(53066),Y=i(82548),W=i(39883),J=i(79876),X=i(26524),ee=i(32048),ne=i.n(ee),ie=i(12674),le=function(e,n){var i=S()(e.created_at),l=S()(n.created_at);return i.diff(l)},te=function(){var e=(0,b.$G)().t,n=A.Z.useToken().token,l=(0,c.Dj)(),s=(0,ie.s0)(),d=(0,ie.UO)().serviceId,g=(0,c.Kr)("initial-fetch"),p=(0,r.Z)(g,2),f=p[0],h=p[1],x=(0,k.useTransition)(),y=(0,r.Z)(x,2),j=y[0],K=y[1],P=(0,k.useTransition)(),R=(0,r.Z)(P,2),ee=R[0],te=R[1],re=(0,k.useState)(null),ae=(0,r.Z)(re,2),oe=ae[0],se=ae[1],de=(0,k.useState)(!1),ue=(0,r.Z)(de,2),ce=ue[0],me=ue[1],ge=(0,k.useState)(!1),ve=(0,r.Z)(ge,2),pe=ve[0],fe=ve[1],he=(0,k.useState)({current:1,pageSize:100}),xe=(0,r.Z)(he,1)[0],ye=(0,C.useLazyLoadQuery)(void 0!==t?t:t=i(4464),{tokenListOffset:(xe.current-1)*xe.pageSize,tokenListLimit:xe.pageSize,endpointId:d||""},{fetchPolicy:"initial-fetch"===f?"store-and-network":"network-only",fetchKey:f}),Se=ye.endpoint,ke=ye.endpoint_token_list,be=(0,m.Y)((function(){if(Se)return(0,u.Lc)({method:"POST",url:"/services/".concat(Se.endpoint_id,"/errors/clear"),client:l})})),Ze=function(){var e="default";switch((arguments.length>0&&void 0!==arguments[0]?arguments[0]:"").toUpperCase()){case"HEALTHY":e="success";break;case"PROVISIONING":e="processing";break;case"UNHEALTHY":e="warning"}return e},je=JSON.parse((null===Se||void 0===Se?void 0:Se.resource_opts)||"{}");return(0,Z.jsxs)(v.Z,{direction:"column",align:"stretch",style:{margin:n.marginSM},gap:"sm",children:[(0,Z.jsx)(B.Z,{items:[{title:e("modelService.Services"),onClick:function(e){e.preventDefault(),s("/serving")},href:"/serving"},{title:e("modelService.RoutingInfo")}]}),(0,Z.jsxs)(v.Z,{direction:"row",justify:"between",children:[(0,Z.jsx)(O.Z.Title,{level:3,style:{margin:0},children:(null===Se||void 0===Se?void 0:Se.name)||""}),(0,Z.jsxs)(v.Z,{gap:"xxs",children:[((null===Se||void 0===Se?void 0:Se.retries)||0)>0?(0,Z.jsx)(H.Z,{title:e("modelService.ClearErrors"),children:(0,Z.jsx)(E.ZP,{loading:ee,icon:(0,Z.jsx)(M.Z,{}),onClick:function(){te((function(){be.mutate(void 0,{onSuccess:function(){return K((function(){h()}))}})}))}})}):(0,Z.jsx)(Z.Fragment,{}),(0,Z.jsx)(E.ZP,{loading:j,icon:(0,Z.jsx)(N.Z,{}),onClick:function(){K((function(){h()}))},children:e("button.Refresh")})]})]}),(0,Z.jsx)(Q.Z,{title:e("modelService.ServiceInfo"),extra:(0,Z.jsx)(E.ZP,{type:"primary",icon:(0,Z.jsx)(D.Z,{}),disabled:((null===Se||void 0===Se?void 0:Se.desired_session_count)||0)<0,onClick:function(){me(!0)},children:e("button.Edit")}),children:(0,Z.jsx)(L.Z,{bordered:!0,column:{xxl:3,xl:3,lg:2,md:1,sm:1,xs:1},style:{backgroundColor:n.colorBgBase},items:[{label:e("modelService.EndpointName"),children:(0,Z.jsx)(O.Z.Text,{copyable:!0,children:null===Se||void 0===Se?void 0:Se.name})},{label:e("modelService.Status"),children:(0,Z.jsx)(o.Z,{endpointFrgmt:Se})},{label:e("modelService.EndpointId"),children:null===Se||void 0===Se?void 0:Se.endpoint_id},{label:e("modelService.SessionOwner"),children:l.email||""},{label:e("modelService.DesiredSessionCount"),children:null===Se||void 0===Se?void 0:Se.desired_session_count},{label:e("modelService.ServiceEndpoint"),children:null!==Se&&void 0!==Se&&Se.url?(0,Z.jsx)(O.Z.Text,{copyable:!0,children:null===Se||void 0===Se?void 0:Se.url}):(0,Z.jsx)(Y.Z,{children:e("modelService.NoServiceEndpoint")})},{label:e("modelService.OpenToPublic"),children:null!==Se&&void 0!==Se&&Se.open_to_public?(0,Z.jsx)(V.Z,{}):(0,Z.jsx)(U.Z,{})},{label:e("modelService.resources"),children:(0,Z.jsxs)(v.Z,{direction:"row",wrap:"wrap",gap:"md",children:[(0,Z.jsx)(H.Z,{title:e("session.ResourceGroup"),children:(0,Z.jsx)(Y.Z,{children:null===Se||void 0===Se?void 0:Se.resource_group})}),ne().map(JSON.parse((null===Se||void 0===Se?void 0:Se.resource_slots)||"{}"),(function(e,n){return(0,Z.jsx)(T.ZP,{type:n,value:e,opts:je},n)}))]}),span:{xl:2}},{label:e("session.launcher.ModelStorage"),children:(0,Z.jsx)(k.Suspense,{fallback:(0,Z.jsx)(W.Z,{indicator:(0,Z.jsx)(q.Z,{spin:!0})}),children:(null===Se||void 0===Se?void 0:Se.model)&&(0,Z.jsx)(G,{uuid:null===Se||void 0===Se?void 0:Se.model,clickable:!1})})},{label:e("modelService.Image"),children:(null===Se||void 0===Se?void 0:Se.image)&&(0,Z.jsxs)(v.Z,{direction:"row",gap:"xs",children:[(0,Z.jsx)(F.Z,{image:Se.image}),(0,Z.jsx)(a.default,{children:Se.image})]}),span:{xl:2}}]})}),(0,Z.jsx)(Q.Z,{title:e("modelService.GeneratedTokens"),extra:(0,Z.jsx)(E.ZP,{type:"primary",icon:(0,Z.jsx)(z.Z,{}),onClick:function(){fe(!0)},children:e("modelService.GenerateToken")}),children:(0,Z.jsx)(J.Z,{scroll:{x:"max-content"},rowKey:"token",columns:[{title:"#",fixed:"left",render:function(e,n,i){return++i},showSorterTooltip:!1},{title:"Token",dataIndex:"token",fixed:"left",render:function(e,n){return(0,Z.jsx)(O.Z.Text,{ellipsis:!0,copyable:!0,style:{width:150},children:n.token})}},{title:"Status",render:function(e,n){var i=S().utc(n.valid_until).isBefore();return(0,Z.jsx)(Y.Z,{color:i?"red":"green",children:i?"Expired":"Valid"})}},{title:"Valid Until",dataIndex:"valid_until",render:function(e,n){return(0,Z.jsx)("span",{children:n.valid_until?S().utc(n.valid_until).tz().format("ll LTS"):"-"})},defaultSortOrder:"descend",sortDirections:["descend","ascend","descend"],sorter:le},{title:"Created at",dataIndex:"created_at",render:function(e,n){return(0,Z.jsx)("span",{children:S()(n.created_at).format("ll LT")})},defaultSortOrder:"descend",sortDirections:["descend","ascend","descend"],sorter:le}],pagination:!1,dataSource:(0,u.uU)(null===ke||void 0===ke?void 0:ke.items),bordered:!0})}),(0,Z.jsx)(Q.Z,{title:e("modelService.RoutesInfo"),children:(0,Z.jsx)(J.Z,{scroll:{x:"max-content"},columns:[{title:e("modelService.RouteId"),dataIndex:"routing_id",fixed:"left"},{title:e("modelService.SessionId"),dataIndex:"session"},{title:e("modelService.Status"),render:function(e,i){var l=i.session,t=i.status;return t&&(0,Z.jsxs)(Z.Fragment,{children:[(0,Z.jsx)(Y.Z,{color:Ze(t),style:{marginRight:0},children:t.toUpperCase()},t),"FAILED_TO_START"===t&&(0,Z.jsx)(X.Z,{children:(0,Z.jsx)(E.ZP,{size:"small",type:"text",icon:(0,Z.jsx)($.Z,{}),style:{color:n.colorTextSecondary},onClick:function(){return l&&function(e){if(null!==Se){var n=Se.errors.find((function(n){var i=n.session_id;return e===i}));se(n||null)}}(l)}})})]})}},{title:e("modelService.TrafficRatio"),dataIndex:"traffic_ratio"}],pagination:!1,dataSource:null===Se||void 0===Se?void 0:Se.routings,rowKey:"routing_id",bordered:!0})}),(0,Z.jsx)(I,{open:!!oe,inferenceSessionErrorFrgmt:oe,onRequestClose:function(){return se(null)}}),(0,Z.jsx)(w.Z,{open:ce,onRequestClose:function(e){me(!1),e&&K((function(){h()}))},endpointFrgmt:Se}),(0,Z.jsx)(_,{open:pe,onRequestClose:function(e){fe(!1),e&&K((function(){h()}))},endpoint_id:(null===Se||void 0===Se?void 0:Se.endpoint_id)||""})]})}},4464:function(e,n,i){i.r(n);var l=function(){var e={defaultValue:null,kind:"LocalArgument",name:"endpointId"},n={defaultValue:null,kind:"LocalArgument",name:"tokenListLimit"},i={defaultValue:null,kind:"LocalArgument",name:"tokenListOffset"},l={kind:"Variable",name:"endpoint_id",variableName:"endpointId"},t=[l],r={alias:null,args:null,kind:"ScalarField",name:"name",storageKey:null},a={alias:null,args:null,kind:"ScalarField",name:"endpoint_id",storageKey:null},o={alias:null,args:null,kind:"ScalarField",name:"image",storageKey:null},s={alias:null,args:null,kind:"ScalarField",name:"desired_session_count",storageKey:null},d={alias:null,args:null,kind:"ScalarField",name:"url",storageKey:null},u={alias:null,args:null,kind:"ScalarField",name:"open_to_public",storageKey:null},c={alias:null,args:null,kind:"ScalarField",name:"session_id",storageKey:null},m={alias:null,args:null,kind:"ScalarField",name:"retries",storageKey:null},g={alias:null,args:null,kind:"ScalarField",name:"model",storageKey:null},v={alias:null,args:null,kind:"ScalarField",name:"model_mount_destiation",storageKey:null},p={alias:null,args:null,kind:"ScalarField",name:"resource_group",storageKey:null},f={alias:null,args:null,kind:"ScalarField",name:"resource_slots",storageKey:null},h={alias:null,args:null,kind:"ScalarField",name:"resource_opts",storageKey:null},x={alias:null,args:null,kind:"ScalarField",name:"routing_id",storageKey:null},y={alias:null,args:null,kind:"ScalarField",name:"session",storageKey:null},S={alias:null,args:null,kind:"ScalarField",name:"traffic_ratio",storageKey:null},k={alias:null,args:null,kind:"ScalarField",name:"endpoint",storageKey:null},b={alias:null,args:null,kind:"ScalarField",name:"status",storageKey:null},Z={alias:null,args:null,kind:"ScalarField",name:"id",storageKey:null},j={alias:null,args:[l,{kind:"Variable",name:"limit",variableName:"tokenListLimit"},{kind:"Variable",name:"offset",variableName:"tokenListOffset"}],concreteType:"EndpointTokenList",kind:"LinkedField",name:"endpoint_token_list",plural:!1,selections:[{alias:null,args:null,kind:"ScalarField",name:"total_count",storageKey:null},{alias:null,args:null,concreteType:"EndpointToken",kind:"LinkedField",name:"items",plural:!0,selections:[Z,{alias:null,args:null,kind:"ScalarField",name:"token",storageKey:null},a,{alias:null,args:null,kind:"ScalarField",name:"domain",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"project",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"session_owner",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"created_at",storageKey:null},{alias:null,args:null,kind:"ScalarField",name:"valid_until",storageKey:null}],storageKey:null}],storageKey:null};return{fragment:{argumentDefinitions:[e,n,i],kind:"Fragment",metadata:null,name:"RoutingListPageQuery",selections:[{alias:null,args:t,concreteType:"Endpoint",kind:"LinkedField",name:"endpoint",plural:!1,selections:[r,a,o,s,d,u,{alias:null,args:null,concreteType:"InferenceSessionError",kind:"LinkedField",name:"errors",plural:!0,selections:[c,{args:null,kind:"FragmentSpread",name:"ServingRouteErrorModalFragment"}],storageKey:null},m,g,v,p,f,h,{alias:null,args:null,concreteType:"Routing",kind:"LinkedField",name:"routings",plural:!0,selections:[x,y,S,k,b],storageKey:null},{args:null,kind:"FragmentSpread",name:"EndpointStatusTagFragment"},{args:null,kind:"FragmentSpread",name:"ModelServiceSettingModal_endpoint"}],storageKey:null},j],type:"Queries",abstractKey:null},kind:"Request",operation:{argumentDefinitions:[e,i,n],kind:"Operation",name:"RoutingListPageQuery",selections:[{alias:null,args:t,concreteType:"Endpoint",kind:"LinkedField",name:"endpoint",plural:!1,selections:[r,a,o,s,d,u,{alias:null,args:null,concreteType:"InferenceSessionError",kind:"LinkedField",name:"errors",plural:!0,selections:[c,{alias:null,args:null,concreteType:"InferenceSessionErrorInfo",kind:"LinkedField",name:"errors",plural:!0,selections:[{alias:null,args:null,kind:"ScalarField",name:"repr",storageKey:null}],storageKey:null}],storageKey:null},m,g,v,p,f,h,{alias:null,args:null,concreteType:"Routing",kind:"LinkedField",name:"routings",plural:!0,selections:[x,y,S,k,b,Z],storageKey:null},Z,b],storageKey:null},j]},params:{cacheID:"dcb559f96ebdb88ea48eb267aae5adf8",id:null,metadata:{},name:"RoutingListPageQuery",operationKind:"query",text:"query RoutingListPageQuery(\n  $endpointId: UUID!\n  $tokenListOffset: Int!\n  $tokenListLimit: Int!\n) {\n  endpoint(endpoint_id: $endpointId) {\n    name\n    endpoint_id\n    image\n    desired_session_count\n    url\n    open_to_public\n    errors {\n      session_id\n      ...ServingRouteErrorModalFragment\n    }\n    retries\n    model\n    model_mount_destiation\n    resource_group\n    resource_slots\n    resource_opts\n    routings {\n      routing_id\n      session\n      traffic_ratio\n      endpoint\n      status\n      id\n    }\n    ...EndpointStatusTagFragment\n    ...ModelServiceSettingModal_endpoint\n    id\n  }\n  endpoint_token_list(offset: $tokenListOffset, limit: $tokenListLimit, endpoint_id: $endpointId) {\n    total_count\n    items {\n      id\n      token\n      endpoint_id\n      domain\n      project\n      session_owner\n      created_at\n      valid_until\n    }\n  }\n}\n\nfragment EndpointStatusTagFragment on Endpoint {\n  id\n  status\n}\n\nfragment ModelServiceSettingModal_endpoint on Endpoint {\n  endpoint_id\n  desired_session_count\n}\n\nfragment ServingRouteErrorModalFragment on InferenceSessionError {\n  session_id\n  errors {\n    repr\n  }\n}\n"}}}();l.hash="4c9244fddb4b985f78d843419d34f832",n.default=l}}]);
//# sourceMappingURL=232.5714de5c.chunk.js.map