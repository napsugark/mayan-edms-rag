'use strict';

$.fn.hasAnyClass = function() {
    /*
     *  Return true is an element has any of the passed classes
     *  The classes are passed as an array.
     */
    for (const cssClass of arguments[0]) {
        if (this.hasClass(cssClass)) {
            return true;
        }
    }
    return false;
}

class PartialNavigation {
    constructor (parameters) {
        parameters = parameters || {};

        // lastLocation - used as the AJAX referer
        this.lastLocation = null;

        // initialURL - the URL to send users when trying to access the / URL
        this.initialURL = parameters.initialURL || null;

        // disabledAnchorClasses - Anchors with any of these classes will not be
        // processes as AJAX anchors and their events nulled
        this.disabledAnchorClasses = parameters.disabledAnchorClasses || [];

        // excludeAnchorClasses - Anchors with any of these classes will not be
        // processes as AJAX anchors
        this.excludeAnchorClasses = parameters.excludeAnchorClasses || [];

        this.redirectionCode = parameters.redirectionCode;

        if (!this.redirectionCode) {
            alert('Need to setup redirectionCode');
        }

        if (!this.initialURL) {
            alert('Need to setup initialURL');
        }

        // AJAX request throttling and pending request cancellation.
        // Default is 10 requests in 5 seconds of less.
        this.maximumAjaxRequests = parameters.maximumAjaxRequests || 10;
        this.ajaxRequestTimeout = parameters.ajaxRequestTimeout || 5000;
        this.ajaxThrottlingMessage = parameters.ajaxThrottlingMessage || 'Too many requests.';

        this.currentAjaxRequest = null;
        this.AjaxRequestTimeOutList = [];

        // AJAX Refresh button.
        this.ajaxRefreshButtonAnimationSpeed = 1000;
        this.ajaxRefreshButtonEnabled = true;
        this.ajaxRefreshButtonTimer = setTimeout(null);

        this.$ajaxContent = $('#ajax-content');
    }

    initialize () {
        this.setupAjaxAnchors();
        this.setupAjaxNavigation();
        this.setupAjaxForm();
        this.setupAjaxRefreshButton();
    }

    ajaxContentSet (content) {
        const app = this;
        const htmlContent = app.$ajaxContent.html();

        app.$ajaxContent.trigger('preupdate');
        app.$ajaxContent.html(content).ready(function () {
            app.$ajaxContent.trigger('updated');
        });

        return htmlContent;
    }

    filterLocation (newLocation) {
        /*
         * Method to validate new locations
         */
        let url;

        try {
            url = new URL(newLocation, window.location.origin);
        } catch (error) {
            if (error instanceof TypeError) {
                return this.initialURL;
            } else {
                throw error;
            }
        }

        if (url.pathname === '/') {
            // href with no path remain in the same location
            // We strip the same location query and use the new href's one.
            const urlNew = new URL(window.location.hash.substring(1), url);
            urlNew.search = newLocation;

            if (urlNew.pathname === '/') {
                return this.initialURL;
            } else {
                return `${urlNew.pathname}${urlNew.search}`;
            }
        }

        return newLocation;
    }

    loadAjaxContent (url) {
        /*
         *  Method to load and display partial backend views to the main
         *  view port.
         */
        const app = this;

        url = this.filterLocation(url);

        this.AjaxRequestTimeOutList.push(
            setTimeout(function() {
                app.AjaxRequestTimeOutList.shift();
            }, app.ajaxRequestTimeout)
        );

        // Request exceeded maximum, ignoring.
        if (this.AjaxRequestTimeOutList.length > app.maximumAjaxRequests) {
            let options = {};

            options['timeOut'] = 10000;
            $('body').css('cursor', 'progress');

            toastr['warning'](app.ajaxThrottlingMessage, '', options);
            return;
        }

        // Another AJAX request is being processed. Cancel the previous
        // one.
        if (this.currentAjaxRequest) {
            // Store and repaint the content area to avoid a '0' status
            // server error message.
            const htmlContent = app.ajaxContentSet();

            this.currentAjaxRequest.abort();
            $('body').css('cursor', 'progress');

            app.ajaxContentSet(htmlContent);
        }

        this.currentAjaxRequest = $.ajax({
            async: true,
            dataType: 'html',
            error: function (jqXHR, textStatus, errorThrown) {
                app.processAjaxRequestError(jqXHR);
            },
            // Need to set mimeType only when run from local file.
            mimeType: 'text/html; charset=utf-8',
            success: function (data, textStatus, response) {
                if (response.status == app.redirectionCode) {
                    // Handle redirects.
                    const newLocation = response.getResponseHeader('Location');

                    app.setLocation(newLocation);
                    app.lastLocation = newLocation;
                } else {
                    app.lastLocation = url;
                    if (response.getResponseHeader('Content-Disposition')) {
                        window.location = this.url;
                    } else {
                        app.ajaxContentSet(data);
                        $('body').css('cursor', 'default');
                    }
                }

                // Enable requests again.
                app.currentAjaxRequest = null;

                // Reset throttling.
                for (let item of app.AjaxRequestTimeOutList) {
                    clearTimeout(item);
                }
                app.AjaxRequestTimeOutList = [];
            },
            type: 'GET',
            url: url
        });
    }

    onAnchorClick ($this, event) {
        /*
         * Anchor click event manager. We intercept all click events and
         * route them to load the content via AJAX instead.
         */
        if ($this.hasAnyClass(this.excludeAnchorClasses)) {
            return true;
        }

        if ($this.hasAnyClass(this.disabledAnchorClasses)) {
            event.preventDefault();
            return;
        }

        if ($this.parents().hasAnyClass(this.disabledAnchorClasses)) {
            return false;
        }

        const url = $this.attr('href');
        if (url === undefined) {
            return true;
        }

        if (url.indexOf('javascript:;') > -1) {
            // Ignore/exclude links meant to execute javascript on click.
            return true;
        }

        if (url === '#') {
            // Ignore/exclude links with only a hash.
            return true;
        }

        event.preventDefault();

        if (event.ctrlKey) {
            window.open(url);
            return false;
        }

        if (!($this.hasClass('disabled') || $this.parent().hasClass('disabled'))) {
            this.setLocation(url);
        }
    }

    processAjaxRequestError (jqXHR) {
        /*
         * Method to process an AJAX request and make it presentable to the
         * user.
         */
        const app = this;

        if (djangoDEBUG) {
            let errorMessage = null;

            if (jqXHR.status != 0) {
                errorMessage = jqXHR.responseText || jqXHR.statusText;
            } else {
                errorMessage = 'Server communication error.';
            }

            app.ajaxContentSet(
                ` \
                    <div class="row">\
                        <div class="col-xs-12">\
                            <div id="banner-server-error">\
                                <div class="alert alert-danger" role="alert"><i class="fa fa-exclamation-triangle"></i> Server error, status code: ${jqXHR.status}</div> \
                                    <pre id="django-server-error"><code>${errorMessage}</code> \
                                    </pre> \
                                </div>\
                            </div>\
                    </div>\
                `
            );

            // Call Django's debug view initial JavaScript.
            if (jqXHR.status === 500) {
                  hideAll(document.querySelectorAll('table.vars'));
                  hideAll(document.querySelectorAll('ol.pre-context'));
                  hideAll(document.querySelectorAll('ol.post-context'));
                  hideAll(document.querySelectorAll('div.pastebin'));
            }
        } else {
            if (jqXHR.status === 0) {
                if (jqXHR.statusText !== "abort") {
                    $('#modal-server-error .modal-body').html($('#template-error').html());
                    $('#modal-server-error').modal('show')
                }
            } else {
                if ([403, 404, 500].indexOf(jqXHR.status !== -1)) {
                    app.ajaxContentSet(jqXHR.responseText);
                } else {
                    app.ajaxContentSet(jqXHR.statusText);
                }
            }
        }
    }

    setLocation (newLocation, pushState) {
        /*
         * Method to update the browsers history and trigger a page update.
         */

        // Validate the new location first.
        newLocation = this.filterLocation(newLocation);

        if (typeof pushState === 'undefined') {
            // Check if we should just load the content or load the content
            // and update the history.
            pushState = true;
        }

        const urlNew = new URL(window.location);
        urlNew.hash = newLocation;

        if (pushState) {
            history.pushState({}, '', urlNew);
        }
        this.loadAjaxContent(newLocation);
    }

    async setupAjaxAnchors () {
        /*
         * Setup the new click event handler.
         */
        const app = this;
        $('body').on('click', 'a', function (event) {
            app.onAnchorClick($(this), event);
        });
    }

    async setupAjaxForm () {
        /*
         * Method to setup the handling of form in an AJAX way.
         */
        const app = this;
        let lastAjaxFormData = {};

        $('form').ajaxForm({
            async: true,
            beforeSubmit: function(arr, $form, options) {
                const urlDefault = new URL(
                    window.location.hash.substring(1), window.location
                );
                const stringFormAction = $form.attr('action') || urlDefault.toString();

                options.url = stringFormAction;

                const urlSearchParamForm = new URLSearchParams(
                    decodeURIComponent($form.serialize())
                );
                const urlFormAction = new URL(stringFormAction, window.location);

                urlFormAction.search = urlSearchParamForm.toString();
                lastAjaxFormData.url = urlFormAction;

                if ($form.attr('target') == '_blank') {
                    // If the form has a target attribute we emulate it by
                    // opening a new window and passing the form serialized
                    // data as the query.
                    window.open(urlFormAction.toString());

                    return false;
                }
            },
            dataType: 'html',
            delegation: true,
            error: function(jqXHR, textStatus, errorThrown){
                app.processAjaxRequestError(jqXHR);
            },
            // ! Need set mimeType only when run from local file.
            mimeType: 'text/html; charset=utf-8',
            success: function(data, textStatus, request) {
                if (request.status == app.redirectionCode) {
                    // Handle redirects after submitting the form.
                    const newLocation = request.getResponseHeader('Location');

                    app.setLocation(newLocation);
                } else {
                    const urlCurrent = new URL(window.location.origin);
                    urlCurrent.hash = `${lastAjaxFormData.url.pathname}${lastAjaxFormData.url.search}`;
                    history.pushState({}, '', urlCurrent);
                    app.ajaxContentSet(data);

                }
            }
        });
    }

    async setupAjaxRefreshButton () {
        const app = this;

        $('body').on('click', 'a.appearance-link-ajax-refresh', function (event) {
            const $this = $(this);

            $this.blur();

            event.preventDefault();

            if (app.ajaxRefreshButtonEnabled) {
                app.ajaxRefreshButtonEnabled = false;

                clearTimeout(app.ajaxRefreshButtonTimer);
                app.setLocation(window.location.hash.substring(1));
                $this.addClass('fa-spin');
                $this.css(
                    'animation-duration',
                    `${app.ajaxRefreshButtonAnimationSpeed}ms`
                );

                app.ajaxRefreshButtonTimer = setTimeout(function () {
                    $this.removeClass('fa-spin');
                    app.ajaxRefreshButtonEnabled = true;
                }, app.ajaxRefreshButtonAnimationSpeed);
            }
        });
    }

    async setupAjaxNavigation () {
        /*
         * Setup the navigation method using the hash of the location.
         * Also handles the back button event and loads via AJAX any
         * URL in the location when the app first launches. Registers
         * a callback to send an emulated `HTTP_REFERER` so that the backends
         * code will still work without change.
         */
        const app = this;

        // Load AJAX content when the hash changes.
        if (window.history && window.history.pushState) {
            $(window).on('popstate', function() {
                app.setLocation(window.location.hash.substring(1), false);
            });
        }

        // Load any initial address in the URL of the browser.
        if (window.location.hash) {
            this.setLocation(window.location.hash.substring(1));
        } else {
            this.setLocation('/');
        }

        $.ajaxSetup({
            beforeSend: function (jqXHR, settings) {
                // Emulate the `HTTP_REFERER`.
                jqXHR.setRequestHeader('X-Alt-Referer', app.lastLocation);
            },
        });
    }
}
