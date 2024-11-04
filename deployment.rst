==============================
Data Collector xApp Deployment
==============================

.. code-block:: bash

    cd ~/oaic
    git clone https://github.com/wchen2654/dc-xapp.git

    cd ~/oaic/dc-xapp
    sudo cp dc-config-file.json /var/www/xApp_config.local/config_files/
    sudo systemctl reload nginx

    sudo docker build . -t xApp-registry.local:5008/dc:0.1.0

Paste the following in the ss-xapp-onboard.url file located in the ss-xapp directory. Substitute the <machine_ip_addr> with the IP address of your machine. You can find this by pasting the command hostname -I | cut -f1 -d' ' in the terminal.

.. code-block::bash

    vim dc-xapp-onboard.url

Paste the following in url file. **Remember to change Ip address**

.. code-block::bash

    {"config-file.json_url":"http://<machine_ip_addr>:5010/config_files/dc-config-file.json"}

Deploying the xApp
==================

.. code-block::bash

    cd ~/oaic/dc-xapp

.. code-block::bash

    export KONG_PROXY=`sudo kubectl get svc -n ricplt -l app.kubernetes.io/name=kong -o jsonpath='{.items[0].spec.clusterIP}'`
    export E2MGR_HTTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-e2mgr-http -o jsonpath='{.items[0].spec.clusterIP}'`
    export APPMGR_HTTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-appmgr-http -o jsonpath='{.items[0].spec.clusterIP}'`
    export E2TERM_SCTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-e2term-sctp-alpha -o jsonpath='{.items[0].spec.clusterIP}'`
    export ONBOARDER_HTTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-xapp-onboarder-http -o jsonpath='{.items[0].spec.clusterIP}'`
    export RTMGR_HTTP=`sudo kubectl get svc -n ricplt --field-selector metadata.name=service-ricplt-rtmgr-http -o jsonpath='{.items[0].spec.clusterIP}'`

.. warning::

    If you are repeating an experiement, you may want to restart the pod using the command below. By doing so, you do not have to undeploy and redeploy the xApp again.

.. code-block:: bash

    sudo kubectl -n ricxapp rollout restart deployment ricxapp-dc

Running the xApp
================

.. code-block:: bash

    sudo kubectl logs -f -n ricxapp -l app=ricxapp-dc

.. code-block:: bash

    cd ~/oaic/dc-xapp
    sudo chmod +x zmqtwoue.sh
    sudo ./zmqtwoue.sh
